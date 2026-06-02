"""
finetune_experts.py — Expert-sequential fine-tuning for MIMO V4.

Multi-phase fine-tuning pipeline:
  Phase 1: Freeze encoder + move_head, tune each expert head independently
           with its own loss function. Save best checkpoint per head.
  Phase 2: Load best expert weights, freeze everything except move_head,
           tune only move_logits with CE loss.

The encoder (board_encoder, tabular_encoder, move_encoder, cross-attention,
trunk) is frozen throughout all phases.

CLI flags --fix-{wdl,mistake,time-spent,move}-head skip individual phases.
--fix-move-head skips Phase 2 entirely.

Usage:
  # Full pipeline — tune all 3 experts then move head:
  python finetune_experts.py --checkpoint best_model.pt --data-dir dataset/v4

  # Skip WDL expert, tune mistake + time + move head:
  python finetune_experts.py --checkpoint best_model.pt --data-dir dataset/v4 --fix-wdl-head

  # Only tune mistake expert (for parallel runs on different GPUs):
  python finetune_experts.py --checkpoint best_model.pt --data-dir dataset/v4 \
      --fix-wdl-head --fix-time-spent-head --fix-move-head

  # Only tune move head (experts already tuned):
  python finetune_experts.py --checkpoint best_model.pt --data-dir dataset/v4 \
      --fix-wdl-head --fix-mistake-head --fix-time-spent-head \
      --best-mistake-ckpt mistake_best.pt --best-time-ckpt time_best.pt --best-wdl-ckpt wdl_best.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from chess_mimo_model_v4 import ChessMIMOModelV4
from mimo_dataset_polars import MIMOCompactDataset, ShardGroupSampler


# ---------------------------------------------------------------------------
# Parameter-group helpers
# ---------------------------------------------------------------------------

ENCODER_PREFIXES = (
    'board_encoder.', 'tabular_encoder.', 'move_encoder.',
    'query_proj.', 'cross_attn.', 'attn_norm.', 'attn_gate.',
    'trunk.',
)

EXPERT_MAP = {
    'mistake':    'mistake_expert.',
    'time_spent': 'time_expert.',
    'wdl':        'wdl_before_expert.',
}

MOVE_HEAD_PREFIX = 'move_head.'


def _freeze_all(model: nn.Module):
    """Freeze every parameter in the model."""
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_prefix(model: nn.Module, prefix: str):
    """Unfreeze parameters whose name starts with prefix."""
    for name, p in model.named_parameters():
        if name.startswith(prefix):
            p.requires_grad = True


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Single-head loss functions
# ---------------------------------------------------------------------------

def mistake_loss_fn(predictions: Dict, targets: Dict) -> torch.Tensor:
    """BCE loss for mistake head, masked to valid moves."""
    valid_mask = targets['move_idx'] >= 0
    raw_bce = F.binary_cross_entropy_with_logits(
        predictions['mistake_prob'].squeeze(-1),
        targets['is_mistake'], reduction='none',
    )
    masked = raw_bce * valid_mask.float()
    return masked.sum() / valid_mask.float().sum().clamp(min=1.0)


def time_loss_fn(predictions: Dict, targets: Dict) -> torch.Tensor:
    """Huber loss for time spent head."""
    return F.huber_loss(
        predictions['time_spent'].squeeze(-1),
        targets['time_spent_log'], delta=2.0,
    )


def wdl_loss_fn(predictions: Dict, targets: Dict) -> torch.Tensor:
    """Cross-entropy (KL) loss for WDL head."""
    log_pred = torch.log(predictions['win_prob_before'].clamp(min=1e-8))
    return -(targets['win_prob_before'] * log_pred).sum(-1).mean()


def move_loss_fn(predictions: Dict, targets: Dict) -> torch.Tensor:
    """Cross-entropy loss for move logits."""
    return F.cross_entropy(
        predictions['move_logits'], targets['move_idx'], ignore_index=-1,
    )


HEAD_LOSS_FNS = {
    'mistake':    mistake_loss_fn,
    'time_spent': time_loss_fn,
    'wdl':        wdl_loss_fn,
}

# Head-specific accuracy metrics
def mistake_accuracy(predictions: Dict, targets: Dict) -> float:
    pred = (predictions['mistake_prob'].squeeze(-1) > 0.0).float()
    return (pred == targets['is_mistake']).float().mean().item()


def time_mae(predictions: Dict, targets: Dict) -> float:
    pred = predictions['time_spent'].squeeze(-1)
    return torch.abs(pred - targets['time_spent_log']).mean().item()


def wdl_accuracy(predictions: Dict, targets: Dict) -> float:
    pred = predictions['win_prob_before'].argmax(dim=-1)
    true = targets['win_prob_before'].argmax(dim=-1)
    return (pred == true).float().mean().item()


HEAD_METRIC_FNS = {
    'mistake':    ('mistake_acc', mistake_accuracy),
    'time_spent': ('time_mae', time_mae),
    'wdl':        ('wdl_acc', wdl_accuracy),
}


# ---------------------------------------------------------------------------
# Data loading (reused across phases)
# ---------------------------------------------------------------------------

def build_loaders(args) -> Tuple[DataLoader, DataLoader, Optional[ShardGroupSampler]]:
    """Build train and val DataLoaders from --data-dir."""
    if args.data_dir:
        train_path = os.path.join(args.data_dir, 'train')
        val_path = os.path.join(args.data_dir, 'val')
    else:
        train_path = args.train_data
        val_path = args.val_data

    train_ds = MIMOCompactDataset(train_path, max_possible=args.max_possible,
                                   cache_shards=args.cache_shards)
    val_ds = MIMOCompactDataset(val_path, max_possible=args.max_possible,
                                 cache_shards=args.cache_shards)

    # Shard-aware sampler: shuffle shards then within-shard for near-100% cache hits
    train_sampler = None
    if hasattr(train_ds, 'shard_counts') and train_ds.shard_counts:
        train_sampler = ShardGroupSampler(
            train_ds.shard_offsets, train_ds.shard_counts)
        print(f"[DATA] Using ShardGroupSampler ({len(train_ds.shard_counts)} shards)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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
    return train_loader, val_loader, train_sampler


def _batch_to_device(batch: Dict, device: torch.device) -> Tuple[Dict, Dict]:
    """Move batch to device and split into model inputs + targets."""
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
    """Load a V4 checkpoint, handling _orig_mod. prefix from torch.compile."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Read model config from checkpoint if available
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

    # Handle _orig_mod. prefix from torch.compile
    state = ckpt.get('model_state_dict', ckpt)
    cleaned = {}
    for k, v in state.items():
        clean_k = k.replace('_orig_mod.', '')
        cleaned[clean_k] = v
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
# Training loop (generic — works for any head or move)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: Optional[GradScaler],
    head_name: str,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, elapsed_seconds)."""
    model.train()
    running_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for batch in loader:
        inputs, targets = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            outputs = model(*inputs)
            loss = loss_fn(outputs, targets)

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
        n_batches += 1

        if n_batches % 200 == 0:
            avg = running_loss / n_batches
            elapsed = time.time() - t0
            print(f"  [{head_name}] batch {n_batches}/{len(loader)} "
                  f"loss={avg:.4f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    return running_loss / max(n_batches, 1), elapsed


@torch.no_grad()
def validate_head(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    metric_fn,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    """Validate for one head. Returns (avg_loss, metric_value)."""
    model.eval()
    running_loss = 0.0
    metric_sum = 0.0
    n_batches = 0

    for batch in loader:
        inputs, targets = _batch_to_device(batch, device)

        with autocast('cuda', enabled=use_amp):
            outputs = model(*inputs)
            loss = loss_fn(outputs, targets)

        running_loss += loss.item()
        if metric_fn is not None:
            metric_sum += metric_fn(outputs, targets)
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    avg_metric = metric_sum / max(n_batches, 1)
    return avg_loss, avg_metric


@torch.no_grad()
def validate_move(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, float, float]:
    """Validate move head. Returns (loss, top1, top3, top5)."""
    model.eval()
    running_loss = 0.0
    correct1 = correct3 = correct5 = 0
    total = 0
    n_batches = 0

    for batch in loader:
        inputs, targets = _batch_to_device(batch, device)

        with autocast('cuda', enabled=use_amp):
            outputs = model(*inputs)
            loss = move_loss_fn(outputs, targets)

        running_loss += loss.item()
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

    avg_loss = running_loss / max(n_batches, 1)
    return (avg_loss,
            correct1 / max(total, 1),
            correct3 / max(total, 1),
            correct5 / max(total, 1))


# ---------------------------------------------------------------------------
# Phase 1: Expert head tuning
# ---------------------------------------------------------------------------

def tune_expert_head(
    model: ChessMIMOModelV4,
    head_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
    output_dir: Path,
    train_sampler=None,
) -> Optional[Path]:
    """
    Tune a single expert head. Returns path to best checkpoint, or None.

    Freezes everything, unfreezes only the target expert module,
    trains with the head-specific loss, saves best by val loss.
    """
    prefix = EXPERT_MAP[head_name]
    loss_fn = HEAD_LOSS_FNS[head_name]
    metric_name, metric_fn = HEAD_METRIC_FNS[head_name]

    print(f"\n{'='*60}")
    print(f"  PHASE 1: Tuning {head_name} expert")
    print(f"{'='*60}")

    # Freeze all, unfreeze only this expert
    _freeze_all(model)
    _unfreeze_prefix(model, prefix)
    trainable = _count_trainable(model)
    print(f"  Trainable params: {trainable:,} ({prefix}*)")

    # Optimizer for only the unfrozen params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.expert_lr, weight_decay=args.weight_decay)

    # Cosine schedule
    total_steps = args.expert_epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.expert_lr, total_steps=total_steps,
        pct_start=warmup_steps / total_steps if total_steps > 0 else 0.05,
        anneal_strategy='cos',
    )

    use_amp = args.amp and device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    best_ckpt_path = output_dir / f'best_{head_name}_expert.pt'
    patience_counter = 0

    for epoch in range(1, args.expert_epochs + 1):
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
        train_loss, elapsed = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            use_amp, scaler, head_name,
        )
        # Step scheduler per batch is handled inside OneCycleLR
        # But we created it with total_steps, so we need per-batch stepping
        # Actually OneCycleLR needs step() per batch. Let me fix this.
        # For simplicity, use per-epoch scheduling instead.

        val_loss, val_metric = validate_head(
            model, val_loader, loss_fn, metric_fn, device, use_amp)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            # Save only the expert module weights
            expert_state = {
                k: v for k, v in model.state_dict().items()
                if k.startswith(prefix)
            }
            torch.save({
                'head': head_name,
                'prefix': prefix,
                'epoch': epoch,
                'val_loss': val_loss,
                'expert_state_dict': expert_state,
            }, best_ckpt_path)
        else:
            patience_counter += 1

        # Epoch-by-epoch checkpoint
        epoch_state = {
            k: v for k, v in model.state_dict().items()
            if k.startswith(prefix)
        }
        epoch_ckpt_path = output_dir / f'{head_name}_expert_epoch_{epoch}.pt'
        torch.save({
            'head': head_name,
            'prefix': prefix,
            'epoch': epoch,
            'val_loss': val_loss,
            'expert_state_dict': epoch_state,
        }, epoch_ckpt_path)

        marker = '★' if is_best else ' '
        lower_better = head_name == 'time_spent'
        metric_label = metric_name
        print(f"  [{head_name}] epoch {epoch}/{args.expert_epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"{metric_label}={val_metric:.4f}  ({elapsed:.0f}s) {marker}")

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"  [{head_name}] Early stopping after {args.patience} epochs without improvement")
            break

    print(f"  [{head_name}] Best val_loss={best_val_loss:.4f} → {best_ckpt_path.name}")
    return best_ckpt_path


# ---------------------------------------------------------------------------
# Phase 2: Move head tuning
# ---------------------------------------------------------------------------

def tune_move_head(
    model: ChessMIMOModelV4,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
    output_dir: Path,
    expert_ckpts: Dict[str, Path],
    train_sampler=None,
) -> Optional[Path]:
    """
    Tune only the move head after loading best expert weights.

    Freezes everything, loads best expert checkpoints, then unfreezes
    only move_head and trains with CE loss.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Tuning move head")
    print(f"{'='*60}")

    # Load best expert weights into model
    for head_name, ckpt_path in expert_ckpts.items():
        if ckpt_path and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            expert_state = ckpt['expert_state_dict']
            # Partial load — only update the expert's keys
            model_state = model.state_dict()
            model_state.update(expert_state)
            model.load_state_dict(model_state)
            print(f"  Loaded {head_name} expert from {ckpt_path.name} "
                  f"(epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?'):.4f})")

    # Freeze all, unfreeze only move_head
    _freeze_all(model)
    _unfreeze_prefix(model, MOVE_HEAD_PREFIX)
    trainable = _count_trainable(model)
    print(f"  Trainable params: {trainable:,} ({MOVE_HEAD_PREFIX}*)")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.move_lr, weight_decay=args.weight_decay)

    use_amp = args.amp and device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    best_ckpt_path = output_dir / 'best_move_head.pt'
    patience_counter = 0

    for epoch in range(1, args.move_epochs + 1):
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch + 1000)  # offset to avoid same shuffle as Phase 1
        train_loss, elapsed = train_one_epoch(
            model, train_loader, move_loss_fn, optimizer, device,
            use_amp, scaler, 'move',
        )

        val_loss, top1, top3, top5 = validate_move(
            model, val_loader, device, use_amp)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'head': 'move',
                'epoch': epoch,
                'val_loss': val_loss,
                'top1': top1, 'top3': top3, 'top5': top5,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_ckpt_path)
        else:
            patience_counter += 1

        # Epoch-by-epoch checkpoint
        epoch_ckpt_path = output_dir / f'move_head_epoch_{epoch}.pt'
        torch.save({
            'head': 'move',
            'epoch': epoch,
            'val_loss': val_loss,
            'top1': top1, 'top3': top3, 'top5': top5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, epoch_ckpt_path)

        marker = '★' if is_best else ' '
        print(f"  [move] epoch {epoch}/{args.move_epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"top1={top1:.3f}  top3={top3:.3f}  top5={top5:.3f}  "
              f"({elapsed:.0f}s) {marker}")

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"  [move] Early stopping after {args.patience} epochs without improvement")
            break

    print(f"  [move] Best val_loss={best_val_loss:.4f} → {best_ckpt_path.name}")
    return best_ckpt_path


# ---------------------------------------------------------------------------
# Assemble final model
# ---------------------------------------------------------------------------

def assemble_final(
    model: ChessMIMOModelV4,
    expert_ckpts: Dict[str, Path],
    move_ckpt: Optional[Path],
    device: torch.device,
    output_dir: Path,
):
    """
    Assemble the final model from the best expert + move checkpoints
    and save a complete checkpoint.
    """
    print(f"\n{'='*60}")
    print(f"  ASSEMBLING FINAL MODEL")
    print(f"{'='*60}")

    model_state = model.state_dict()

    # Load expert weights
    for head_name, ckpt_path in expert_ckpts.items():
        if ckpt_path and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model_state.update(ckpt['expert_state_dict'])
            print(f"  ← {head_name} expert (val_loss={ckpt.get('val_loss', '?'):.4f})")

    # Load move head weights (full model state from move phase)
    if move_ckpt and move_ckpt.exists():
        ckpt = torch.load(move_ckpt, map_location=device, weights_only=False)
        move_state = ckpt.get('model_state_dict', {})
        # Only take move_head keys
        for k, v in move_state.items():
            if k.startswith(MOVE_HEAD_PREFIX):
                model_state[k] = v
        print(f"  ← move head (val_loss={ckpt.get('val_loss', '?'):.4f}, "
              f"top1={ckpt.get('top1', '?'):.3f})")

    model.load_state_dict(model_state)

    final_path = output_dir / 'expert_tuned_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'expert_ckpts': {k: str(v) for k, v in expert_ckpts.items() if v},
        'move_ckpt': str(move_ckpt) if move_ckpt else None,
        'model_config': {
            'max_possible': model.max_possible,
            'hidden_dim': model.hidden_dim,
            'component_versions': model.component_versions,
        },
    }, final_path)
    print(f"  → {final_path} ({final_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Expert-sequential fine-tuning for MIMO V4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument('--checkpoint', required=True, help='V4 base checkpoint to fine-tune from')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Dataset root with train/val/test subdirs')
    parser.add_argument('--train-data', type=str, default=None, help='Legacy: train shard path')
    parser.add_argument('--val-data', type=str, default=None, help='Legacy: val shard path')
    parser.add_argument('--output-dir', default='checkpoints/expert_tuning',
                        help='Output directory for checkpoints')

    # Head skipping
    parser.add_argument('--fix-wdl-head', action='store_true',
                        help='Skip WDL expert tuning phase')
    parser.add_argument('--fix-mistake-head', action='store_true',
                        help='Skip mistake expert tuning phase')
    parser.add_argument('--fix-time-spent-head', action='store_true',
                        help='Skip time-spent expert tuning phase')
    parser.add_argument('--fix-move-head', action='store_true',
                        help='Skip move head tuning phase (Phase 2) entirely')

    # Pre-tuned expert checkpoints (for Phase 2 only runs)
    parser.add_argument('--best-mistake-ckpt', type=str, default=None,
                        help='Pre-tuned mistake expert checkpoint (skips its Phase 1)')
    parser.add_argument('--best-time-ckpt', type=str, default=None,
                        help='Pre-tuned time expert checkpoint (skips its Phase 1)')
    parser.add_argument('--best-wdl-ckpt', type=str, default=None,
                        help='Pre-tuned WDL expert checkpoint (skips its Phase 1)')

    # Training hyperparams
    parser.add_argument('--expert-epochs', type=int, default=10,
                        help='Epochs per expert head (Phase 1)')
    parser.add_argument('--move-epochs', type=int, default=10,
                        help='Epochs for move head (Phase 2)')
    parser.add_argument('--expert-lr', type=float, default=1e-4,
                        help='Learning rate for expert heads')
    parser.add_argument('--move-lr', type=float, default=3e-4,
                        help='Learning rate for move head')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (0=disabled)')

    # Model
    parser.add_argument('--max-possible', type=int, default=220)
    parser.add_argument('--cnn-channels', type=int, default=128)
    parser.add_argument('--res-blocks', type=int, default=6)
    parser.add_argument('--hidden-dim', type=int, default=256)
    # Component versions (registry) — None = inherit from checkpoint
    parser.add_argument('--mistake-expert-ver', type=str, default=None,
                        help='Swap mistake expert to this registry version before tuning')
    parser.add_argument('--time-expert-ver', type=str, default=None,
                        help='Swap time expert to this registry version before tuning')
    parser.add_argument('--wdl-expert-ver', type=str, default=None,
                        help='Swap WDL expert to this registry version before tuning')
    parser.add_argument('--move-head-ver', type=str, default=None,
                        help='Swap move head to this registry version before Phase 2')

    # System
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--cache-shards', type=int, default=2,
                        help='Number of shards to keep in LRU cache per worker')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use mixed precision (default: on)')
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch.compile (may interfere with partial freezing)')

    args = parser.parse_args()

    # Validate
    if not args.data_dir and not (args.train_data and args.val_data):
        parser.error("Provide --data-dir or both --train-data and --val-data")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model
    model = load_checkpoint(args.checkpoint, device, args)

    # Runtime expert/move_head swaps (fresh weights — will be tuned below)
    _swap_map = {'mistake': args.mistake_expert_ver, 'time': args.time_expert_ver,
                 'wdl': args.wdl_expert_ver}
    for head_name, ver in _swap_map.items():
        if ver is not None and ver != model._expert_versions.get(head_name):
            model.swap_expert(head_name, ver)
    if args.move_head_ver is not None and args.move_head_ver != model._move_head_version:
        model.swap_move_head(args.move_head_ver)

    train_loader, val_loader, train_sampler = build_loaders(args)

    # Determine which experts to tune
    expert_schedule = []
    if not args.fix_mistake_head and not args.best_mistake_ckpt:
        expert_schedule.append('mistake')
    if not args.fix_time_spent_head and not args.best_time_ckpt:
        expert_schedule.append('time_spent')
    if not args.fix_wdl_head and not args.best_wdl_ckpt:
        expert_schedule.append('wdl')

    # Collect best checkpoints (pre-provided or to-be-tuned)
    expert_ckpts: Dict[str, Optional[Path]] = {
        'mistake':    Path(args.best_mistake_ckpt) if args.best_mistake_ckpt else None,
        'time_spent': Path(args.best_time_ckpt) if args.best_time_ckpt else None,
        'wdl':        Path(args.best_wdl_ckpt) if args.best_wdl_ckpt else None,
    }

    # ── Phase 1: Expert tuning ──
    if expert_schedule:
        print(f"\n[PLAN] Phase 1: Tuning experts: {expert_schedule}")
        for head_name in expert_schedule:
            ckpt_path = tune_expert_head(
                model, head_name, train_loader, val_loader,
                args, device, output_dir, train_sampler=train_sampler,
            )
            expert_ckpts[head_name] = ckpt_path
    else:
        print("\n[PLAN] Phase 1: Skipped (all experts fixed or pre-provided)")

    # ── Phase 2: Move head tuning ──
    move_ckpt = None
    if not args.fix_move_head:
        print(f"\n[PLAN] Phase 2: Tuning move head")
        move_ckpt = tune_move_head(
            model, train_loader, val_loader, args, device,
            output_dir, expert_ckpts, train_sampler=train_sampler,
        )
    else:
        print("\n[PLAN] Phase 2: Skipped (--fix-move-head)")

    # ── Assemble final model ──
    assemble_final(model, expert_ckpts, move_ckpt, device, output_dir)

    # Save run config
    config_path = output_dir / 'expert_tuning_config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"\n✓ Expert tuning complete. Output: {output_dir}")


if __name__ == '__main__':
    main()
