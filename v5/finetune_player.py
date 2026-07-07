"""
finetune_player.py — Player-replication fine-tuning for MIMO V5.

Traditional fine-tuning that takes a V5 base checkpoint and tunes it on
games from a specific player to replicate their playing style.

V5 additions:
  - Uses MIMOLossV5 (5 heads including contrastive) for Kendall weighting.
  - Checkpoint loading handles V5 config: contrastive, phase-gated experts,
    attention move head.
  - Phase encoder and contrastive modules can optionally be frozen.

Training strategy:
  - Phase 1 (optional): Freeze encoder, tune only heads for
    --freeze-encoder-epochs to adapt outputs without disturbing representations.
  - Phase 2: Unfreeze everything, fine-tune end-to-end with a lower LR.
  - Cosine annealing with warmup, gradient clipping, early stopping.

Usage:
  python finetune_player.py --checkpoint best_model.pt \
      --data-dir dataset/player_magnus/ --player-name "Magnus Carlsen"

  python finetune_player.py --checkpoint best_model.pt \
      --data-dir dataset/player_magnus/ --player-name "Magnus Carlsen" \
      --freeze-encoder-epochs 3
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

from chess_mimo_model_v5 import ChessMIMOModelV5, MIMOLossV5
from mimo_dataset_polars import MIMOCompactDataset


# ---------------------------------------------------------------------------
# Parameter groups
# ---------------------------------------------------------------------------

ENCODER_PREFIXES = (
    'board_encoder.', 'tabular_encoder.', 'move_encoder.',
    'query_proj.', 'cross_attn.', 'attn_norm.', 'attn_gate.',
    'trunk.',
    # V5: contrastive and phase modules are encoder-level
    'contrastive_encoder.', 'contrastive_anchor_proj.',
    'phase_encoder.',
)


def _is_encoder_param(name: str) -> bool:
    return any(name.startswith(p) for p in ENCODER_PREFIXES)


def _freeze_experts(model: nn.Module):
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
        persistent_workers=args.workers > 0, drop_last=True,
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
    cp = cp.to(memory_format=torch.channels_last)
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

def load_checkpoint(checkpoint_path: str, device: torch.device, args) -> ChessMIMOModelV5:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('model_config', ckpt.get('config', {}))
    comp_ver = config.get('component_versions', {})

    # Component version overrides: CLI wins, then checkpoint, then 'default'
    mistake_ver = getattr(args, 'mistake_expert_ver', None) or comp_ver.get('mistake_expert', 'default')
    time_ver = getattr(args, 'time_expert_ver', None) or comp_ver.get('time_expert', 'default')
    wdl_ver = getattr(args, 'wdl_expert_ver', None) or comp_ver.get('wdl_expert', 'default')
    mh_ver = getattr(args, 'move_head_ver', None) or comp_ver.get('move_head', 'default')
    contrastive_dim = getattr(args, 'contrastive_embed_dim', None)

    # ---- Resolve architecture from checkpoint ----
    cli_preset = getattr(args, 'preset', None)
    ckpt_preset = config.get('effective_preset')
    eff = dict(config.get('effective_config', {}) or {})

    if cli_preset:
        preset_name = cli_preset
        print(f"[LOAD] Architecture from CLI --preset: {preset_name}")
    elif ckpt_preset and ckpt_preset in ChessMIMOModelV5.PRESETS:
        preset_name = ckpt_preset
        print(f"[LOAD] Architecture from checkpoint preset: {preset_name}")
    else:
        preset_name = None

    if preset_name:
        preset_overrides = {
            'max_possible': args.max_possible,
            'tabular_dim': 20, 'move_scalar_dim': 13,
            'mistake_expert_ver': mistake_ver, 'time_expert_ver': time_ver,
            'wdl_expert_ver': wdl_ver, 'move_head_ver': mh_ver,
            'num_phases': getattr(args, 'num_phases', 3) or 3,
        }
        if contrastive_dim is not None:
            preset_overrides['contrastive_embed_dim'] = contrastive_dim
        if config.get('tactical_preprocessor_config'):
            preset_overrides['tactical_preprocessor_config'] = config['tactical_preprocessor_config']
        model = ChessMIMOModelV5.from_preset(preset_name, **preset_overrides)
    elif eff:
        print(f"[LOAD] Architecture from checkpoint effective_config")
        def _arch(default, *keys):
            for src in (eff, config):
                for k in keys:
                    v = src.get(k)
                    if v is not None:
                        return v
            return default
        if contrastive_dim is None:
            contrastive_dim = _arch(64, 'contrastive_embed_dim')
        model = ChessMIMOModelV5(
            max_possible=_arch(args.max_possible, 'max_possible'),
            cnn_channels=_arch(args.cnn_channels, 'cnn_channels'),
            num_res_blocks=_arch(args.res_blocks, 'num_res_blocks', 'res_blocks'),
            hidden_dim=_arch(args.hidden_dim, 'hidden_dim'),
            tabular_dim=_arch(20, 'tabular_dim'),
            move_scalar_dim=_arch(13, 'move_scalar_dim'),
            sq_embed_dim=_arch(48, 'sq_embed_dim'),
            expert_hidden=_arch(160, 'expert_hidden'),
            mistake_expert_ver=mistake_ver, time_expert_ver=time_ver,
            wdl_expert_ver=wdl_ver, move_head_ver=mh_ver,
            contrastive_embed_dim=contrastive_dim,
            contrastive_hidden_dim=_arch(128, 'contrastive_hidden_dim'),
            contrastive_margin=_arch(1.0, 'contrastive_margin'),
            use_phase_experts=_arch(True, 'use_phase_experts'),
            phase_hidden_dim=_arch(64, 'phase_hidden_dim'),
            num_phases=_arch(3, 'num_phases'),
            use_film=_arch(True, 'use_film'),
            film_hidden_dim=_arch(64, 'film_hidden_dim'),
            use_tactical_enrichment=_arch(False, 'use_tactical_enrichment'),
            tactical_preprocessor_config=config.get('tactical_preprocessor_config'),
        )
    else:
        print(f"[LOAD] WARNING: No architecture info in checkpoint — using CLI defaults!")
        if contrastive_dim is None:
            contrastive_dim = 64
        model = ChessMIMOModelV5(
            max_possible=args.max_possible, cnn_channels=args.cnn_channels,
            num_res_blocks=args.res_blocks, hidden_dim=args.hidden_dim,
            mistake_expert_ver=mistake_ver, time_expert_ver=time_ver,
            wdl_expert_ver=wdl_ver, move_head_ver=mh_ver,
            contrastive_embed_dim=contrastive_dim,
        )

    state = ckpt.get('model_state_dict', ckpt)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[V5] {len(missing)} missing keys (new V5 modules with fresh weights)")
    model.to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    epoch = ckpt.get('epoch', 0)
    val_loss = ckpt.get('val_loss', float('inf'))
    print(f"[LOAD] Checkpoint: {checkpoint_path}")
    print(f"[LOAD] Epoch {epoch}, val_loss={val_loss:.4f}")
    print(f"[LOAD] Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[LOAD] Versions: {model.component_versions}")
    print(f"[LOAD] V5: contrastive_dim={model.contrastive_embed_dim}, "
          f"phase_experts={model.use_phase_experts}")

    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, use_amp, scaler, epoch):
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
            # Fix #6: skip non-finite loss/grad batches to avoid poisoning weights
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

        running_loss += loss.item()
        for k, v in loss_dict.items():
            running_comp[k] = running_comp.get(k, 0.0) + v
        n_batches += 1

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
def validate(model, loader, criterion, device, use_amp):
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

        mpred = (outputs['mistake_prob'].squeeze(-1) > 0.0).float()
        mistake_correct += (mpred == targets['is_mistake']).float().sum().item()
        mistake_total += len(targets['is_mistake'])

        if 'win_prob_before' in outputs:
            wp = outputs['win_prob_before'].argmax(dim=-1)
            wt = targets['win_prob_before'].argmax(dim=-1)
            wdl_correct += (wp == wt).sum().item()
            wdl_total += wt.numel()

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
        description='Player-replication fine-tuning for MIMO V5',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument('--checkpoint', required=True, help='V5 base checkpoint')
    parser.add_argument('--preset', type=str, default=None,
                        choices=['v5-minimal', 'v5', 'v5-large', 'v5-widehead'],
                        help='Model preset — overrides architecture defaults. Normally read '
                             'from checkpoint automatically; use only if checkpoint lacks config.')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--train-data', type=str, default=None)
    parser.add_argument('--val-data', type=str, default=None)
    parser.add_argument('--output-dir', default='checkpoints/player_tuning')
    parser.add_argument('--player-name', type=str, default='unknown')

    # Training strategy
    parser.add_argument('--freeze-encoder-epochs', type=int, default=0)
    parser.add_argument('--move-only-pct', type=float, default=0.5)
    parser.add_argument('--move-only-after', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--encoder-lr-scale', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--warmup-pct', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=7)
    # V5: contrastive margin for loss
    parser.add_argument('--contrastive-margin', type=float, default=1.0)

    # Model
    parser.add_argument('--max-possible', type=int, default=220)
    parser.add_argument('--cnn-channels', type=int, default=128)
    parser.add_argument('--res-blocks', type=int, default=6)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--mistake-expert-ver', type=str, default=None)
    parser.add_argument('--time-expert-ver', type=str, default=None)
    parser.add_argument('--wdl-expert-ver', type=str, default=None)
    parser.add_argument('--move-head-ver', type=str, default=None)
    parser.add_argument('--contrastive-embed-dim', type=int, default=None,
                        help='Override contrastive encoder embed dim. None=inherit from checkpoint; 0 disables the contrastive encoder (for the no-contrastive ablation arm).')

    # System
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.add_argument('--compile', action='store_true', default=False)

    args = parser.parse_args()

    if not args.data_dir and not (args.train_data and args.val_data):
        parser.error("Provide --data-dir or both --train-data and --val-data")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Speedup #1: enable TF32 tensor cores for float32 matmul/conv on Ada.
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    player_slug = args.player_name.lower().replace(' ', '_')
    output_dir = Path(args.output_dir) / player_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model + criterion
    model = load_checkpoint(args.checkpoint, device, args)
    criterion = MIMOLossV5(contrastive_margin=args.contrastive_margin).to(device)
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

    param_groups.append({'params': list(criterion.parameters()), 'lr': args.lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay, fused=True)

    use_amp = args.amp and device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None

    trainable = _count_trainable(model) + sum(p.numel() for p in criterion.parameters())
    print(f"[TRAIN] Trainable params: {trainable:,}")
    print(f"[TRAIN] Player: {args.player_name}")
    print(f"[TRAIN] LR: {args.lr} (encoder: "
          f"{args.lr * args.encoder_lr_scale if not encoder_frozen else 'frozen'})")

    best_val_loss = float('inf')
    best_ckpt_path = output_dir / f'best_player_{player_slug}.pt'
    patience_counter = 0
    history = []

    if args.move_only_after > 0:
        move_only_epoch = args.move_only_after
    elif args.move_only_pct > 0:
        move_only_epoch = max(1, int(args.epochs * args.move_only_pct))
    else:
        move_only_epoch = args.epochs + 1
    move_only_phase = False

    print(f"[TRAIN] Move-only phase starts after epoch {move_only_epoch}"
          if move_only_epoch <= args.epochs else "[TRAIN] Move-only phase disabled")

    for epoch in range(1, args.epochs + 1):
        if encoder_frozen and epoch > args.freeze_encoder_epochs:
            _unfreeze_all(model)
            encoder_frozen = False
            param_groups = [
                {'params': encoder_params, 'lr': args.lr * args.encoder_lr_scale},
                {'params': head_params, 'lr': args.lr},
                {'params': list(criterion.parameters()), 'lr': args.lr},
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay, fused=True)
            trainable = _count_trainable(model) + sum(p.numel() for p in criterion.parameters())
            print(f"\n  [E{epoch}] Encoder unfrozen! Trainable: {trainable:,}")

        if not move_only_phase and epoch > move_only_epoch:
            move_only_phase = True
            _freeze_experts(model)
            _freeze_encoder(model)
            move_head_params = [p for p in model.move_head.parameters() if p.requires_grad]
            param_groups = [
                {'params': move_head_params, 'lr': args.lr},
                {'params': list(criterion.parameters()), 'lr': args.lr},
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay, fused=True)
            trainable = _count_trainable(model) + sum(p.numel() for p in criterion.parameters())
            print(f"\n  [E{epoch}] ★ Move-only phase! Trainable: {trainable:,}")

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
                'config': {
                    'max_possible': model.max_possible,
                    'hidden_dim': model.hidden_dim,
                    'component_versions': model.component_versions,
                    'contrastive_embed_dim': model.contrastive_embed_dim,
                    'contrastive_margin': model.contrastive_margin,
                    'use_phase_experts': model.use_phase_experts,
                    'phase_hidden_dim': model.phase_hidden_dim,
                    'num_phases': model.num_phases,
                },
            }, best_ckpt_path)
        else:
            patience_counter += 1

        marker = '★' if is_best else ' '
        frozen_tag = (' [move-only]' if move_only_phase else
                      (' [enc frozen]' if encoder_frozen else ''))

        train_str = f"  TRAIN loss={train_loss:.4f}  move_acc={train_acc:.3f}"
        for h in ['move_logits', 'mistake_prob', 'win_prob_before', 'time_spent', 'contrastive']:
            if h in train_comp:
                train_str += f"  {h}={train_comp[h]:.4f}"

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

        loss_str = '  LOSSES'
        for h in ['move_logits', 'mistake_prob', 'win_prob_before', 'time_spent', 'contrastive']:
            if h in val_comp:
                loss_str += f"  {h}={val_comp[h]:.4f}"
        print(loss_str)

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, **val_metrics,
        })

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n  Early stopping after {args.patience} epochs")
            break

    history_path = output_dir / f'history_{player_slug}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    config_path = output_dir / f'config_{player_slug}.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"\n✓ Player fine-tuning complete.")
    print(f"  Player: {args.player_name}")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {best_ckpt_path}")


if __name__ == '__main__':
    main()
