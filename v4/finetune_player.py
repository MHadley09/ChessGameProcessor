"""
finetune_player.py — Player-replication fine-tuning for MIMO V4.

Traditional fine-tuning that takes a V4 base checkpoint and tunes it on
games from a specific player (and optionally similar players) to replicate
their playing style.

Supports two modes:
  1. --data-dir: pre-built NPZ shards for the target player(s)
  2. --pgn + --player: PGN file filtered to a specific player's games,
     processed on-the-fly (requires the direct_dataset_processor pipeline)

Training strategy:
  - Phase 1 (optional): Freeze encoder, tune only heads (all 4) for
    --freeze-encoder-epochs to adapt outputs without disturbing representations.
  - Phase 2: Unfreeze everything, fine-tune end-to-end with a lower LR.
  - Cosine annealing with warmup, gradient clipping, early stopping.

All performance optimizations retained: NPZ unpacking, pin_memory,
persistent workers, prefetch, mixed precision, torch.compile.

Usage:
  # From pre-built NPZ shards:
  python finetune_player.py --checkpoint best_model.pt \
      --data-dir dataset/player_magnus/ --player-name "Magnus Carlsen"

  # Freeze encoder for first 3 epochs:
  python finetune_player.py --checkpoint best_model.pt \
      --data-dir dataset/player_magnus/ --player-name "Magnus Carlsen" \
      --freeze-encoder-epochs 3

  # Lower LR for careful tuning:
  python finetune_player.py --checkpoint best_model.pt \
      --data-dir dataset/player_magnus/ --player-name "Magnus Carlsen" \
      --lr 5e-5 --epochs 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from chess_mimo_model_v4 import ChessMIMOModelV4, MIMOLoss
from mimo_dataset_polars import MIMOCompactDataset


# ---------------------------------------------------------------------------
# Parameter groups
# ---------------------------------------------------------------------------

ENCODER_PREFIXES = (
    'board_encoder.', 'tabular_encoder.', 'move_encoder.',
    'query_proj.', 'cross_attn.', 'attn_norm.', 'attn_gate.',
    'trunk.',
)


def _is_encoder_param(name: str) -> bool:
    return any(name.startswith(p) for p in ENCODER_PREFIXES)


def _freeze_experts(model: nn.Module):
    """Freeze all expert heads (mistake, time, wdl_before)."""
    for name, p in model.named_parameters():
        if any(name.startswith(pf) for pf in
               ('mistake_expert.', 'time_expert.', 'wdl_before_expert.')):
            p.requires_grad = False


def _freeze_encoder(model: nn.Module):
    for name, p in model.named_parameters():
        if _is_encoder_param(name):
            p.requires_grad = False


def _unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_loaders(args) -> Tuple[DataLoader, DataLoader]:
    if args.data_dir:
        train_path = os.path.join(args.data_dir, 'train')
        val_path = os.path.join(args.data_dir, 'val')
    else:
        train_path = args.train_data
        val_path = args.val_data

    train_ds = MIMOCompactDataset(train_path, max_possible=args.max_possible)
    val_ds = MIMOCompactDataset(val_path, max_possible=args.max_possible)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        prefetch_factor=args.prefetch if args.workers > 0 else None,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        prefetch_factor=args.prefetch if args.workers > 0 else None,
        persistent_workers=args.workers > 0,
    )

    print(f"[DATA] train: {len(train_ds):,} examples → {len(train_loader):,} batches")
    print(f"[DATA] val:   {len(val_ds):,} examples → {len(val_loader):,} batches")
    return train_loader, val_loader


def _batch_to_device(batch: Dict, device: torch.device):
    cp = batch['current_planes'].to(device, non_blocking=True)
    pf = batch['possible_from_sq'].to(device, non_blocking=True)
    pt = batch['possible_to_sq'].to(device, non_blocking=True)
    pp = batch['possible_promo'].to(device, non_blocking=True)
    ps = batch['possible_scalars'].to(device, non_blocking=True)
    pm = batch['possible_mask'].to(device, non_blocking=True)
    tab = batch['tabular'].to(device, non_blocking=True)
    aidx = batch['actual_idx'].to(device, non_blocking=True)

    inputs = (cp, pf, pt, pp, ps, pm, tab, aidx)
    targets = {
        'move_idx':        aidx,
        'is_mistake':      batch['is_mistake'].to(device, non_blocking=True),
        'win_prob_before': batch['win_prob_before'].to(device, non_blocking=True),
        'time_spent_log':  batch['time_spent_log'].to(device, non_blocking=True),
    }
    return inputs, targets


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str, device: torch.device, args) -> ChessMIMOModelV4:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt.get('model_config', ckpt.get('config', {}))

    # Recover component versions from checkpoint (fall back to CLI / defaults)
    comp_ver = config.get('component_versions', {})
    mistake_ver = getattr(args, 'mistake_expert_ver', None) or comp_ver.get('mistake_expert', 'default')
    time_ver = getattr(args, 'time_expert_ver', None) or comp_ver.get('time_expert', 'default')
    wdl_ver = getattr(args, 'wdl_expert_ver', None) or comp_ver.get('wdl_expert', 'default')
    mh_ver = getattr(args, 'move_head_ver', None) or comp_ver.get('move_head', 'default')

    model = ChessMIMOModelV4(
        max_possible=config.get('max_possible', args.max_possible),
        cnn_channels=config.get('cnn_channels', args.cnn_channels),
        num_res_blocks=config.get('res_blocks', args.res_blocks),
        hidden_dim=config.get('hidden_dim', args.hidden_dim),
        mistake_expert_ver=mistake_ver,
        time_expert_ver=time_ver,
        wdl_expert_ver=wdl_ver,
        move_head_ver=mh_ver,
    )

    state = ckpt.get('model_state_dict', ckpt)
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace('_orig_mod.', '')] = v
    model.load_state_dict(cleaned, strict=False)
    model.to(device)

    epoch = ckpt.get('epoch', 0)
    val_loss = ckpt.get('val_loss', ckpt.get('best_val_loss', float('inf')))
    print(f"[LOAD] Checkpoint: {checkpoint_path}")
    print(f"[LOAD] Epoch {epoch}, val_loss={val_loss:.4f}")
    print(f"[LOAD] Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[LOAD] Versions: {model.component_versions}")

    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MIMOLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: Optional[GradScaler],
    epoch: int,
) -> Tuple[float, float, Dict, float]:
    """Train one epoch. Returns (avg_loss, move_acc, component_losses, elapsed)."""
    model.train()
    running_loss = 0.0
    running_comp = {}
    correct = total = 0
    n_batches = 0
    t0 = time.time()

    for batch in loader:
        inputs, targets = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            outputs = model(*inputs)
            loss, loss_dict = criterion(outputs, targets)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

        running_loss += loss.item()
        for k, v in loss_dict.items():
            running_comp[k] = running_comp.get(k, 0.0) + v
        n_batches += 1

        # Move accuracy
        aidx = targets['move_idx']
        preds = outputs['move_logits'].argmax(dim=1)
        valid = aidx >= 0
        correct += (preds[valid] == aidx[valid]).sum().item()
        total += valid.sum().item()

        if n_batches % 500 == 0:
            avg = running_loss / n_batches
            elapsed = time.time() - t0
            print(f"  [E{epoch}] batch {n_batches}/{len(loader)} "
                  f"loss={avg:.4f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    avg_loss = running_loss / max(n_batches, 1)
    avg_comp = {k: v / max(n_batches, 1) for k, v in running_comp.items()}
    move_acc = correct / max(total, 1)
    return avg_loss, move_acc, avg_comp, elapsed


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: MIMOLoss,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, Dict, Dict]:
    """Validate. Returns (avg_loss, metrics_dict, component_losses)."""
    model.eval()
    running_loss = 0.0
    running_comp = {}
    n_batches = 0
    correct1 = correct3 = correct5 = total = 0
    mistake_correct = mistake_total = 0
    wdl_correct = wdl_total = 0
    time_abs_err = time_total = 0

    for batch in loader:
        inputs, targets = _batch_to_device(batch, device)

        with autocast('cuda', enabled=use_amp):
            outputs = model(*inputs)
            loss, loss_dict = criterion(outputs, targets)

        running_loss += loss.item()
        for k, v in loss_dict.items():
            running_comp[k] = running_comp.get(k, 0.0) + v
        n_batches += 1

        # Move
        logits = outputs['move_logits']
        aidx = targets['move_idx']
        valid = aidx >= 0
        if valid.any():
            topk = logits[valid].topk(min(5, logits.shape[1]), dim=1).indices
            tv = aidx[valid]
            correct1 += (topk[:, 0] == tv).sum().item()
            correct3 += (topk[:, :3] == tv.unsqueeze(1)).any(dim=1).sum().item()
            correct5 += (topk == tv.unsqueeze(1)).any(dim=1).sum().item()
            total += valid.sum().item()

        # Mistake
        mpred = (outputs['mistake_prob'].squeeze(-1) > 0.0).float()
        mistake_correct += (mpred == targets['is_mistake']).float().sum().item()
        mistake_total += len(targets['is_mistake'])

        # WDL
        if 'win_prob_before' in outputs:
            wp = outputs['win_prob_before'].argmax(dim=-1)
            wt = targets['win_prob_before'].argmax(dim=-1)
            wdl_correct += (wp == wt).sum().item()
            wdl_total += wt.numel()

        # Time
        if 'time_spent' in outputs:
            tp = outputs['time_spent'].squeeze(-1)
            time_abs_err += torch.abs(tp - targets['time_spent_log']).sum().item()
            time_total += targets['time_spent_log'].numel()

    avg_loss = running_loss / max(n_batches, 1)
    avg_comp = {k: v / max(n_batches, 1) for k, v in running_comp.items()}
    metrics = {
        'move_top1': correct1 / max(total, 1),
        'move_top3': correct3 / max(total, 1),
        'move_top5': correct5 / max(total, 1),
        'mistake_acc': mistake_correct / max(mistake_total, 1),
        'wdl_acc': wdl_correct / max(wdl_total, 1),
        'time_mae': time_abs_err / max(time_total, 1),
    }
    return avg_loss, metrics, avg_comp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Player-replication fine-tuning for MIMO V4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument('--checkpoint', required=True, help='V4 base checkpoint')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Dataset root with train/val/test NPZ shards for target player(s)')
    parser.add_argument('--train-data', type=str, default=None, help='Legacy: train path')
    parser.add_argument('--val-data', type=str, default=None, help='Legacy: val path')
    parser.add_argument('--output-dir', default='checkpoints/player_tuning',
                        help='Output directory')
    parser.add_argument('--player-name', type=str, default='unknown',
                        help='Player name (used for output naming and metadata)')

    # Training strategy
    parser.add_argument('--freeze-encoder-epochs', type=int, default=0,
                        help='Freeze encoder for first N epochs, then unfreeze (0=never freeze)')
    parser.add_argument('--move-only-pct', type=float, default=0.5,
                        help='Fraction of epochs after which only move head trains (0=disabled, '
                             '0.5=second half is move-only)')
    parser.add_argument('--move-only-after', type=int, default=0,
                        help='Explicit epoch to start move-only phase (overrides --move-only-pct, 0=use pct)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Peak learning rate (lower than base training)')
    parser.add_argument('--encoder-lr-scale', type=float, default=0.1,
                        help='LR multiplier for encoder params after unfreeze (default: 0.1x)')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--warmup-pct', type=float, default=0.1,
                        help='Warmup fraction of total steps')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience (0=disabled)')

    # Model
    parser.add_argument('--max-possible', type=int, default=220)
    parser.add_argument('--cnn-channels', type=int, default=128)
    parser.add_argument('--res-blocks', type=int, default=6)
    parser.add_argument('--hidden-dim', type=int, default=256)
    # Component versions (registry)
    parser.add_argument('--mistake-expert-ver', type=str, default=None,
                        help='Override mistake expert version (default: from checkpoint)')
    parser.add_argument('--time-expert-ver', type=str, default=None,
                        help='Override time expert version')
    parser.add_argument('--wdl-expert-ver', type=str, default=None,
                        help='Override WDL expert version')
    parser.add_argument('--move-head-ver', type=str, default=None,
                        help='Override move head version')

    # System
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='torch.compile the model')

    args = parser.parse_args()

    if not args.data_dir and not (args.train_data and args.val_data):
        parser.error("Provide --data-dir or both --train-data and --val-data")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    player_slug = args.player_name.lower().replace(' ', '_')
    output_dir = Path(args.output_dir) / player_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    model = load_checkpoint(args.checkpoint, device, args)
    criterion = MIMOLoss().to(device)
    train_loader, val_loader = build_loaders(args)

    if args.compile:
        model = torch.compile(model)
        print("[MODEL] torch.compile enabled")

    # Optimizer with differential LR
    encoder_params = []
    head_params = []
    for name, p in model.named_parameters():
        if _is_encoder_param(name):
            encoder_params.append(p)
        else:
            head_params.append(p)

    # Start with encoder frozen if requested
    encoder_frozen = args.freeze_encoder_epochs > 0
    if encoder_frozen:
        _freeze_encoder(model)
        param_groups = [{'params': head_params, 'lr': args.lr}]
        print(f"[TRAIN] Encoder frozen for first {args.freeze_encoder_epochs} epochs")
    else:
        param_groups = [
            {'params': encoder_params, 'lr': args.lr * args.encoder_lr_scale},
            {'params': head_params, 'lr': args.lr},
        ]

    # Include criterion params (Kendall log_vars)
    param_groups.append({'params': list(criterion.parameters()), 'lr': args.lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    use_amp = args.amp and device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None

    trainable = _count_trainable(model) + sum(p.numel() for p in criterion.parameters())
    print(f"[TRAIN] Trainable params: {trainable:,}")
    print(f"[TRAIN] Player: {args.player_name}")
    print(f"[TRAIN] LR: {args.lr} (encoder: {args.lr * args.encoder_lr_scale if not encoder_frozen else 'frozen'})")

    best_val_loss = float('inf')
    best_ckpt_path = output_dir / f'best_player_{player_slug}.pt'
    patience_counter = 0
    history = []

    # Move-only phase: default = second half of epochs
    if args.move_only_after > 0:
        move_only_epoch = args.move_only_after
    elif args.move_only_pct > 0:
        move_only_epoch = max(1, int(args.epochs * args.move_only_pct))
    else:
        move_only_epoch = args.epochs + 1  # disabled
    move_only_phase = False

    print(f"[TRAIN] Move-only phase starts after epoch {move_only_epoch}"
          if move_only_epoch <= args.epochs else "[TRAIN] Move-only phase disabled")

    for epoch in range(1, args.epochs + 1):
        # Unfreeze encoder after freeze period
        if encoder_frozen and epoch > args.freeze_encoder_epochs:
            _unfreeze_all(model)
            encoder_frozen = False
            # Rebuild optimizer with encoder params at lower LR
            param_groups = [
                {'params': encoder_params, 'lr': args.lr * args.encoder_lr_scale},
                {'params': head_params, 'lr': args.lr},
                {'params': list(criterion.parameters()), 'lr': args.lr},
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
            trainable = _count_trainable(model) + sum(p.numel() for p in criterion.parameters())
            print(f"\n  [E{epoch}] Encoder unfrozen! Trainable: {trainable:,}, "
                  f"encoder LR={args.lr * args.encoder_lr_scale:.1e}")

        # Transition to move-only phase
        if not move_only_phase and epoch > move_only_epoch:
            move_only_phase = True
            _freeze_experts(model)
            _freeze_encoder(model)
            # Only move_head params + criterion remain trainable
            move_head_params = [p for p in model.move_head.parameters() if p.requires_grad]
            param_groups = [
                {'params': move_head_params, 'lr': args.lr},
                {'params': list(criterion.parameters()), 'lr': args.lr},
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
            trainable = _count_trainable(model) + sum(p.numel() for p in criterion.parameters())
            print(f"\n  [E{epoch}] ★ Move-only phase! Encoder+experts frozen. "
                  f"Trainable: {trainable:,}")

        train_loss, train_acc, train_comp, elapsed = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_amp, scaler, epoch,
        )

        val_loss, val_metrics, val_comp = validate(
            model, val_loader, criterion, device, use_amp)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'player_name': args.player_name,
                'model_config': {
                    'max_possible': model.max_possible,
                    'hidden_dim': model.hidden_dim,
                    'component_versions': model.component_versions,
                },
            }, best_ckpt_path)
        else:
            patience_counter += 1

        marker = '★' if is_best else ' '
        frozen_tag = ' [move-only]' if move_only_phase else (' [enc frozen]' if encoder_frozen else '')

        # Train summary
        train_str = f"  TRAIN loss={train_loss:.4f}  move_acc={train_acc:.3f}"
        for h in ['move_logits', 'mistake_prob', 'win_prob_before', 'time_spent']:
            if h in train_comp:
                train_str += f"  {h}={train_comp[h]:.4f}"

        # Val summary
        val_str = (f"  VAL   loss={val_loss:.4f}  "
                   f"top1={val_metrics['move_top1']:.3f}  "
                   f"top3={val_metrics['move_top3']:.3f}  "
                   f"top5={val_metrics['move_top5']:.3f}  "
                   f"mistake={val_metrics['mistake_acc']:.3f}  "
                   f"wdl={val_metrics['wdl_acc']:.3f}  "
                   f"time_mae={val_metrics['time_mae']:.3f}")

        print(f"\nEpoch {epoch}/{args.epochs}  ({elapsed:.0f}s){frozen_tag} {marker}")
        print(train_str)
        print(val_str)

        # Per-head val losses
        loss_str = '  LOSSES'
        for h in ['move_logits', 'mistake_prob', 'win_prob_before', 'time_spent']:
            if h in val_comp:
                loss_str += f"  {h}={val_comp[h]:.4f}"
        print(loss_str)

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, **val_metrics,
        })

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n  Early stopping after {args.patience} epochs without improvement")
            break

    # Save history
    history_path = output_dir / f'history_{player_slug}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save config
    config_path = output_dir / f'config_{player_slug}.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"\n✓ Player fine-tuning complete.")
    print(f"  Player: {args.player_name}")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {best_ckpt_path}")
    print(f"  History: {history_path}")


if __name__ == '__main__':
    main()
