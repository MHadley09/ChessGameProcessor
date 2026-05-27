#!/usr/bin/env python3
"""
train_mimo.py — Training script for the MIMO chess model V2 (specialist-expert architecture).

Features:
    - Sharded dataset support (train/val/test directories of .npz shards)
    - Backward-compatible with single .npz files
    - On-the-fly FEN → plane construction (no pre-stored planes)
    - Mixed-precision (AMP) training for RTX 4090 efficiency
    - Cosine annealing with linear warmup
    - Learnable uncertainty-based multi-task loss weighting
    - Gradient clipping
    - Single forward pass with built-in masking (no double forward pass)
    - TensorBoard logging
    - Proper checkpoint saving (best + latest + periodic)
    - Reproducible with --seed

Usage (sharded — preferred):
    python train_mimo.py \
        --data-dir dataset/opus \
        --output-dir checkpoints/run1 \
        --epochs 30 --batch-size 512 --lr 3e-4

Usage (legacy single-file):
    python train_mimo.py \
        --train-data data/train.npz \
        --val-data   data/val.npz \
        --output-dir checkpoints/run1
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from chess_mimo_model_v2 import ChessMIMOModelV2, MIMOLoss
from mimo_dataset_polars import MIMOCompactDataset, ShardGroupSampler, dynamic_collate

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


# ---------------------------------------------------------------------------
# Warmup + cosine schedule
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            scale = self._step / max(self.warmup_steps, 1)
        else:
            progress = (self._step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']


# ---------------------------------------------------------------------------
# Resolve dataset paths
# ---------------------------------------------------------------------------

def resolve_data_paths(args):
    """Return (train_path, val_path) — either directories or single .npz files."""
    if args.data_dir:
        data_dir = Path(args.data_dir)
        train_path = data_dir / 'train'
        val_path = data_dir / 'val'
        if not train_path.is_dir():
            raise FileNotFoundError(f"No train/ subdirectory in {data_dir}")
        if not val_path.is_dir():
            raise FileNotFoundError(f"No val/ subdirectory in {data_dir}")
        return str(train_path), str(val_path)
    elif args.train_data and args.val_data:
        return args.train_data, args.val_data
    else:
        raise ValueError("Provide --data-dir (sharded) or both --train-data and --val-data (legacy)")


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                device, epoch, use_amp, save_every_batches=0, out_dir=None,
                resume_from_batch=0):
    model.train()
    running_loss = 0.0
    running_components = {}
    n_batches = 0
    correct_moves = 0
    total_moves = 0
    t0 = time.time()

    for batch in loader:
        n_batches += 1
        # Skip batches already processed (mid-epoch resume)
        if n_batches <= resume_from_batch:
            scheduler.step()  # keep scheduler in sync
            if n_batches % 10000 == 0:
                print(f"  [RESUME] skipping batch {n_batches:,}/{resume_from_batch:,}...",
                      flush=True)
            continue
        cp = batch['current_planes'].to(device, non_blocking=True)
        pp = batch['possible_planes'].to(device, non_blocking=True)
        ps = batch['possible_scalars'].to(device, non_blocking=True)
        pm = batch['possible_mask'].to(device, non_blocking=True)
        tab = batch['tabular'].to(device, non_blocking=True)
        aidx = batch['actual_idx'].to(device, non_blocking=True)

        targets = {
            'move_idx':        aidx,
            'is_mistake':      batch['is_mistake'].to(device, non_blocking=True),
            'win_prob_before': batch['win_prob_before'].to(device, non_blocking=True),
            'win_prob_after':  batch['win_prob_after'].to(device, non_blocking=True),
            'time_spent_log':  batch['time_spent_log'].to(device, non_blocking=True),
        }

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(cp, pp, ps, pm, tab, actual_idx=aidx)
            loss, loss_dict = criterion(outputs, targets)

        # ---- NaN diagnostic (fires once, then continues) ----
        if not hasattr(train_epoch, '_nan_diagnosed') and (
            torch.isnan(loss) or torch.isinf(loss)
        ):
            train_epoch._nan_diagnosed = True
            print("\n" + "=" * 60, flush=True)
            print("  NaN/Inf DIAGNOSTIC (first occurrence)", flush=True)
            print("=" * 60, flush=True)
            for iname, ival in [('current_planes', cp), ('possible_planes', pp),
                                ('possible_scalars', ps), ('possible_mask', pm),
                                ('tabular', tab), ('actual_idx', aidx)]:
                nans = torch.isnan(ival.float()).sum().item()
                infs = torch.isinf(ival.float()).sum().item()
                print(f"  INPUT  {iname:20s}  shape={str(list(ival.shape)):20s}  "
                      f"dtype={ival.dtype}  nan={nans}  inf={infs}  "
                      f"min={ival.float().min().item():.4f}  max={ival.float().max().item():.4f}",
                      flush=True)
            for oname, oval in outputs.items():
                nans = torch.isnan(oval.float()).sum().item()
                infs = torch.isinf(oval.float()).sum().item()
                print(f"  OUTPUT {oname:20s}  shape={str(list(oval.shape)):20s}  "
                      f"dtype={oval.dtype}  nan={nans}  inf={infs}  "
                      f"min={oval.float().min().item():.4f}  max={oval.float().max().item():.4f}",
                      flush=True)
            for tname, tval in targets.items():
                nans = torch.isnan(tval.float()).sum().item()
                infs = torch.isinf(tval.float()).sum().item()
                print(f"  TARGET {tname:20s}  shape={str(list(tval.shape)):20s}  "
                      f"dtype={tval.dtype}  nan={nans}  inf={infs}  "
                      f"min={tval.float().min().item():.4f}  max={tval.float().max().item():.4f}",
                      flush=True)
            for lname, lval in loss_dict.items():
                print(f"  LOSS   {lname:20s}  = {lval:.6f}", flush=True)
            for pname, pval in criterion.named_parameters():
                print(f"  PARAM  {pname:20s}  = {pval.item():.6f}", flush=True)
            print("=" * 60 + "\n", flush=True)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        running_loss += loss.item()
        for k, v in loss_dict.items():
            running_components[k] = running_components.get(k, 0.0) + v

        # Progress logging every 1000 batches
        if n_batches % 1000 == 0:
            elapsed_sofar = time.time() - t0
            avg_loss_sofar = running_loss / n_batches
            batches_per_sec = n_batches / elapsed_sofar
            print(f"  [TRAIN] batch {n_batches:,}/{len(loader):,} "
                  f"({100*n_batches/len(loader):.1f}%) "
                  f"loss={avg_loss_sofar:.4f} "
                  f"speed={batches_per_sec:.1f} b/s "
                  f"elapsed={elapsed_sofar/60:.1f}m", flush=True)

        # Mid-epoch checkpoint
        if save_every_batches > 0 and n_batches % save_every_batches == 0 and out_dir is not None:
            mid_ckpt = {
                'epoch': epoch, 'batch': n_batches,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_step': scheduler._step,
                'scaler_state_dict': scaler.state_dict(),
                'mid_epoch': True,
            }
            torch.save(mid_ckpt, out_dir / 'mid_epoch.pt')
            print(f"  [CKPT] Mid-epoch checkpoint saved at batch {n_batches:,}", flush=True)

        # Move accuracy
        preds = outputs['move_logits'].argmax(dim=1)
        valid = aidx >= 0
        correct_moves += (preds[valid] == aidx[valid]).sum().item()
        total_moves += valid.sum().item()

    elapsed = time.time() - t0
    avg_loss = running_loss / max(n_batches, 1)
    move_acc = correct_moves / max(total_moves, 1)
    avg_comp = {k: v / max(n_batches, 1) for k, v in running_components.items()}
    return avg_loss, move_acc, avg_comp, elapsed


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    running_components = {}
    n_batches = 0
    correct_top1 = correct_top3 = correct_top5 = 0
    total_moves = 0
    mistake_correct = 0
    mistake_total = 0

    for batch in loader:
        cp = batch['current_planes'].to(device, non_blocking=True)
        pp = batch['possible_planes'].to(device, non_blocking=True)
        ps = batch['possible_scalars'].to(device, non_blocking=True)
        pm = batch['possible_mask'].to(device, non_blocking=True)
        tab = batch['tabular'].to(device, non_blocking=True)
        aidx = batch['actual_idx'].to(device, non_blocking=True)

        targets = {
            'move_idx':        aidx,
            'is_mistake':      batch['is_mistake'].to(device, non_blocking=True),
            'win_prob_before': batch['win_prob_before'].to(device, non_blocking=True),
            'win_prob_after':  batch['win_prob_after'].to(device, non_blocking=True),
            'time_spent_log':  batch['time_spent_log'].to(device, non_blocking=True),
        }

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(cp, pp, ps, pm, tab, actual_idx=aidx)
            loss, loss_dict = criterion(outputs, targets)

        running_loss += loss.item()
        for k, v in loss_dict.items():
            running_components[k] = running_components.get(k, 0.0) + v
        n_batches += 1

        # Move metrics
        logits = outputs['move_logits']
        valid = aidx >= 0
        if valid.any():
            preds_top = logits[valid].topk(min(5, logits.shape[1]), dim=1).indices
            targets_v = aidx[valid]
            correct_top1 += (preds_top[:, 0] == targets_v).sum().item()
            correct_top3 += (preds_top[:, :3] == targets_v.unsqueeze(1)).any(dim=1).sum().item()
            correct_top5 += (preds_top == targets_v.unsqueeze(1)).any(dim=1).sum().item()
            total_moves += valid.sum().item()

        # Mistake accuracy
        mistake_pred = (outputs['mistake_prob'].squeeze(-1) > 0.0).float()  # logits: >0 = prob >0.5
        mistake_correct += (mistake_pred == targets['is_mistake']).float().sum().item()
        mistake_total += len(targets['is_mistake'])

    avg_loss = running_loss / max(n_batches, 1)
    avg_comp = {k: v / max(n_batches, 1) for k, v in running_components.items()}
    metrics = {
        'val_loss': avg_loss,
        'move_top1': correct_top1 / max(total_moves, 1),
        'move_top3': correct_top3 / max(total_moves, 1),
        'move_top5': correct_top5 / max(total_moves, 1),
        'mistake_acc': mistake_correct / max(mistake_total, 1),
    }
    return avg_loss, metrics, avg_comp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train MIMO V2 model')
    # --- Data source (pick one) ---
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Root dataset dir with train/ val/ test/ shard subdirs (preferred)')
    parser.add_argument('--train-data', type=str, default=None,
                        help='Legacy: single .npz or shard directory for training')
    parser.add_argument('--val-data', type=str, default=None,
                        help='Legacy: single .npz or shard directory for validation')
    # --- Training ---
    parser.add_argument('--output-dir', default='checkpoints', help='Output directory')
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch-size', type=int,   default=512)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-pct', type=float, default=0.05, help='Warmup fraction of total steps')
    # --- Model ---
    parser.add_argument('--max-possible', type=int, default=220)
    parser.add_argument('--cnn-channels', type=int, default=128)
    parser.add_argument('--res-blocks', type=int, default=6)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--expert-hidden', type=int, default=128,
                        help='Hidden dimension for expert modules')
    parser.add_argument('--expert-layers', type=int, default=2,
                        help='Number of hidden layers in each expert module')
    # --- Data loading ---
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers (reduced from 8 to avoid shared memory exhaustion on Windows)')
    parser.add_argument('--cache-shards', type=int, default=2,
                        help='Number of shards to keep in LRU cache per worker (mmap = cheap)')
    parser.add_argument('--with-phase', action='store_true', help='Include game phase feature')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='DataLoader prefetch factor per worker')
    # --- Misc ---
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-amp',     action='store_true', help='Disable mixed precision')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    parser.add_argument('--use-ort',    action='store_true',
                        help='Wrap model with ONNX Runtime ORTModule (pip install onnxruntime-training)')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--save-every', type=int, default=1, help='Checkpoint every N epochs')
    parser.add_argument('--save-every-batches', type=int, default=10000,
                        help='Mid-epoch checkpoint every N batches (0 to disable)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g. checkpoints/latest.pt)')
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    use_amp = ('cuda' in args.device) and not args.no_amp
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    train_path, val_path = resolve_data_paths(args)
    print(f"\n[DATA] train: {train_path}")
    print(f"[DATA] val:   {val_path}")

    train_ds = MIMOCompactDataset(train_path, max_possible=args.max_possible,
                                  cache_shards=args.cache_shards, with_phase=args.with_phase)
    val_ds   = MIMOCompactDataset(val_path, max_possible=args.max_possible,
                                  cache_shards=args.cache_shards, with_phase=args.with_phase)

    # Shard-aware sampler: shuffle shards then within-shard for near-100% cache hits
    train_sampler = None
    if hasattr(train_ds, 'shard_offsets') and train_ds.shard_files:
        train_sampler = ShardGroupSampler(
            train_ds.shard_offsets, train_ds.shard_counts, seed=args.seed
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=dynamic_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=dynamic_collate,
    )

    print(f"[DATA] train: {len(train_ds):,} examples → {len(train_loader):,} batches")
    print(f"[DATA] val:   {len(val_ds):,} examples → {len(val_loader):,} batches")

    # ---- Model ----
    model = ChessMIMOModelV2(
        cnn_channels=args.cnn_channels,
        num_res_blocks=args.res_blocks,
        tabular_dim=18,
        max_possible=args.max_possible,
        hidden_dim=args.hidden_dim,
        expert_hidden=args.expert_hidden,
        expert_layers=args.expert_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[MODEL] ChessMIMOModelV2 — {n_params:,} parameters")
    print(f"        CNN channels={args.cnn_channels}, res_blocks={args.res_blocks}, "
          f"hidden={args.hidden_dim}, max_moves={args.max_possible}")
    print(f"        expert_hidden={args.expert_hidden}, expert_layers={args.expert_layers}")

    # ---- ONNX Runtime Training (ORTModule) ----
    if args.use_ort:
        try:
            from onnxruntime.training.ortmodule import ORTModule
            model = ORTModule(model)
            print("[MODEL] ORTModule wrapper enabled (ONNX Runtime Training)")
        except ImportError:
            print("[MODEL] onnxruntime-training not installed, skipping ORTModule")
        except Exception as e:
            print(f"[MODEL] ORTModule failed, continuing without: {e}")

    # ---- torch.compile ----
    if not args.no_compile and not args.use_ort and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("[MODEL] torch.compile(mode='reduce-overhead') enabled")
        except Exception as e:
            print(f"[MODEL] torch.compile failed, continuing without: {e}")

    # ---- Optimiser / scheduler ----
    criterion = MIMOLoss().to(device)
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ---- Resume from checkpoint ----
    start_epoch = 1
    resume_from_batch = 0
    best_val_loss = float('inf')
    history = []
    if args.resume:
        print(f"[RESUME] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        criterion.load_state_dict(ckpt['criterion_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('mid_epoch'):
            # Mid-epoch checkpoint: resume within the same epoch
            start_epoch = ckpt['epoch']
            resume_from_batch = ckpt['batch']
            print(f"[RESUME] Mid-epoch resume: epoch {start_epoch}, batch {resume_from_batch:,}")
        else:
            start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        # Restore scheduler step count
        if 'scheduler_step' in ckpt:
            scheduler._step = ckpt['scheduler_step']
            # Reapply LR from saved step
            for pg, base_lr in zip(scheduler.optimizer.param_groups, scheduler.base_lrs):
                if scheduler._step <= scheduler.warmup_steps:
                    scale = scheduler._step / max(scheduler.warmup_steps, 1)
                else:
                    progress = (scheduler._step - scheduler.warmup_steps) / max(
                        scheduler.total_steps - scheduler.warmup_steps, 1)
                    scale = 0.5 * (1.0 + math.cos(math.pi * progress))
                pg['lr'] = max(scheduler.min_lr, base_lr * scale)
        else:
            # Legacy checkpoint: fast-forward scheduler
            steps_done = ckpt['epoch'] * len(train_loader)
            for _ in range(steps_done):
                scheduler.step()
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        print(f"[RESUME] Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    print(f"[TRAIN] epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, "
          f"warmup={warmup_steps} steps, total={total_steps} steps, AMP={use_amp}")
    print(f"[TRAIN] workers={args.num_workers}, cache_shards={args.cache_shards}, "
          f"prefetch={args.prefetch_factor}")

    # ---- Save config ----
    config = vars(args)
    config['n_params'] = n_params
    config['train_examples'] = len(train_ds)
    config['val_examples'] = len(val_ds)
    config['model_version'] = 'v2'
    with open(out_dir / 'train_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ---- TensorBoard ----
    writer = None
    if HAS_TB:
        writer = SummaryWriter(log_dir=str(out_dir / 'tb'))

    # ---- Training loop ----

    for epoch in range(start_epoch, args.epochs + 1):
        # Update shard sampler epoch for reshuffling
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        train_loss, train_acc, train_comp, elapsed = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, epoch, use_amp,
            save_every_batches=args.save_every_batches, out_dir=out_dir,
            resume_from_batch=resume_from_batch if epoch == start_epoch else 0)
        # Clear mid-epoch resume after first epoch processes
        if epoch == start_epoch:
            resume_from_batch = 0

        val_loss, val_metrics, val_comp = validate(
            model, val_loader, criterion, device, use_amp)

        lr_now = scheduler.lr
        print(f"\nEpoch {epoch}/{args.epochs}  ({elapsed:.0f}s)  lr={lr_now:.2e}")
        print(f"  TRAIN loss={train_loss:.4f}  move_acc={train_acc:.3f}")
        print(f"  VAL   loss={val_loss:.4f}  top1={val_metrics['move_top1']:.3f}  "
              f"top3={val_metrics['move_top3']:.3f}  top5={val_metrics['move_top5']:.3f}  "
              f"mistake_acc={val_metrics['mistake_acc']:.3f}")

        # Task weights
        weights_str = '  '.join(
            f"{k}={v:.3f}" for k, v in val_comp.items() if k.startswith('w_'))
        print(f"  WEIGHTS {weights_str}")

        # Log
        entry = {'epoch': epoch, 'train_loss': train_loss, 'train_move_acc': train_acc,
                 'lr': lr_now, **val_metrics, **{f'train_{k}': v for k, v in train_comp.items()}}
        history.append(entry)

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/move_top1', val_metrics['move_top1'], epoch)
            writer.add_scalar('Acc/move_top3', val_metrics['move_top3'], epoch)
            writer.add_scalar('Acc/mistake', val_metrics['mistake_acc'], epoch)
            writer.add_scalar('LR', lr_now, epoch)
            for k, v in val_comp.items():
                if k.startswith('w_'):
                    writer.add_scalar(f'Weights/{k}', v, epoch)

        # Checkpointing
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_step': scheduler._step,
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'config': config,
        }

        torch.save(ckpt, out_dir / 'latest.pt')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, out_dir / 'best.pt')
            print(f"  ★ New best model (val_loss={val_loss:.4f})")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(ckpt, out_dir / f'epoch_{epoch:03d}.pt')

    # ---- Save history ----
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    if writer:
        writer.close()

    print(f"\n✓ Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {out_dir}")


if __name__ == '__main__':
    main()
