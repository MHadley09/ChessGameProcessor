#!/usr/bin/env python3
"""
train_mimo.py — Training script for the MIMO V5 chess model.

Features:
    - Sharded dataset support (train/val/test directories of .npz shards)
    - Backward-compatible with single .npz files
    - On-the-fly FEN → plane construction (no pre-stored planes)
    - Mixed-precision (AMP) training for RTX 4090 efficiency
    - Cosine annealing with linear warmup
    - Learnable uncertainty-based multi-task loss weighting (5 heads)
    - Gradient clipping
    - Single forward pass with built-in masking (no double forward pass)
    - TensorBoard logging with per-head loss, accuracy, and phase metrics
    - Proper checkpoint saving (best + latest + periodic)
    - Reproducible with --seed

V5 has 5 loss terms: move_logits, mistake_prob, win_prob_before, time_spent, contrastive.
V5 features: ContrastiveEncoder, PhaseGatedExperts, AttentionMoveHead (all toggleable).
Per-epoch reports include per-head losses, accuracies, contrastive metrics, and phase weights.

Usage (sharded — preferred):
    python train_mimo.py \
        --data-dir dataset/opus \
        --output-dir checkpoints/run1 \
        --epochs 30 --batch-size 512 --lr 3e-4

    # With attention move head:
    python train_mimo.py --data-dir dataset/opus --move-head-ver attention

    # Phase experts disabled (contrastive only):
    python train_mimo.py --data-dir dataset/opus --no-phase-experts

    # Contrastive disabled (phase experts only):
    python train_mimo.py --data-dir dataset/opus --contrastive-embed-dim 0
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
from torch.amp import GradScaler, autocast
from mimo_dataset_polars import MIMOCompactDataset, ShardGroupSampler

from chess_mimo_model_v5 import ChessMIMOModelV5, MIMOLossV5

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
                resume_from_batch=0, log_every=500):
    model.train()
    running_loss = 0.0
    running_components = {}
    n_batches = 0
    correct_moves = 0
    total_moves = 0
    # Phase weight accumulators
    phase_weight_sum = None
    phase_weight_count = 0
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
        cp = cp.to(memory_format=torch.channels_last)
        pf = batch['possible_from_sq'].to(device, non_blocking=True)
        pt = batch['possible_to_sq'].to(device, non_blocking=True)
        pp = batch['possible_promo'].to(device, non_blocking=True)
        ps = batch['possible_scalars'].to(device, non_blocking=True)
        pm = batch['possible_mask'].to(device, non_blocking=True)
        tab = batch['tabular'].to(device, non_blocking=True)
        aidx = batch['actual_idx'].to(device, non_blocking=True)

        targets = {
            'move_idx':        aidx,
            'is_mistake':      batch['is_mistake'].to(device, non_blocking=True),
            'win_prob_before': batch['win_prob_before'].to(device, non_blocking=True),
            'time_spent_log':  batch['time_spent_log'].to(device, non_blocking=True),
        }

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(cp, pf, pt, pp, ps, pm, tab, actual_idx=aidx)
            loss, loss_dict = criterion(outputs, targets)

        # ---- NaN diagnostic (fires once, then continues) ----
        if not hasattr(train_epoch, '_nan_diagnosed') and (
            torch.isnan(loss) or torch.isinf(loss)
        ):
            train_epoch._nan_diagnosed = True
            print("\n" + "=" * 60, flush=True)
            print("  NaN/Inf DIAGNOSTIC (first occurrence)", flush=True)
            print("=" * 60, flush=True)
            # Inputs
            for iname, ival in [('current_planes', cp), ('possible_from_sq', pf),
                                ('possible_to_sq', pt), ('possible_promo', pp),
                                ('possible_scalars', ps), ('possible_mask', pm),
                                ('tabular', tab), ('actual_idx', aidx)]:
                nans = torch.isnan(ival.float()).sum().item()
                infs = torch.isinf(ival.float()).sum().item()
                print(f"  INPUT  {iname:20s}  shape={str(list(ival.shape)):20s}  "
                      f"dtype={ival.dtype}  nan={nans}  inf={infs}  "
                      f"min={ival.float().min().item():.4f}  max={ival.float().max().item():.4f}",
                      flush=True)
            # Outputs
            for oname, oval in outputs.items():
                nans = torch.isnan(oval.float()).sum().item()
                infs = torch.isinf(oval.float()).sum().item()
                print(f"  OUTPUT {oname:20s}  shape={str(list(oval.shape)):20s}  "
                      f"dtype={oval.dtype}  nan={nans}  inf={infs}  "
                      f"min={oval.float().min().item():.4f}  max={oval.float().max().item():.4f}",
                      flush=True)
            # Targets
            for tname, tval in targets.items():
                nans = torch.isnan(tval.float()).sum().item()
                infs = torch.isinf(tval.float()).sum().item()
                print(f"  TARGET {tname:20s}  shape={str(list(tval.shape)):20s}  "
                      f"dtype={tval.dtype}  nan={nans}  inf={infs}  "
                      f"min={tval.float().min().item():.4f}  max={tval.float().max().item():.4f}",
                      flush=True)
            # Individual losses
            for lname, lval in loss_dict.items():
                print(f"  LOSS   {lname:20s}  = {lval:.6f}", flush=True)
            # Kendall log_vars
            for pname, pval in criterion.named_parameters():
                print(f"  PARAM  {pname:20s}  = {pval.item():.6f}", flush=True)
            print("=" * 60 + "\n", flush=True)

        if use_amp:
            # NaN-aware batch skip — don't poison model with bad gradients
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                nan_skips = getattr(train_epoch, '_nan_skips', 0) + 1
                train_epoch._nan_skips = nan_skips
                if nan_skips <= 10 or nan_skips % 100 == 0:
                    print(f"  [NaN SKIP] batch {n_batches} — skipped optimizer step "
                          f"(total skips: {nan_skips})", flush=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Fix #6: skip the step if grads are non-finite (would poison weights)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                nan_skips = getattr(train_epoch, '_nan_skips', 0) + 1
                train_epoch._nan_skips = nan_skips
                if nan_skips <= 10 or nan_skips % 100 == 0:
                    print(f"  [GRAD SKIP] batch {n_batches} — non-finite grad norm "
                          f"(total skips: {nan_skips})", flush=True)
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                nan_skips = getattr(train_epoch, '_nan_skips', 0) + 1
                train_epoch._nan_skips = nan_skips
                if nan_skips <= 10 or nan_skips % 100 == 0:
                    print(f"  [NaN SKIP] batch {n_batches} — skipped optimizer step "
                          f"(total skips: {nan_skips})", flush=True)
                continue
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Fix #6: skip the step if grads are non-finite (would poison weights)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                nan_skips = getattr(train_epoch, '_nan_skips', 0) + 1
                train_epoch._nan_skips = nan_skips
                if nan_skips <= 10 or nan_skips % 100 == 0:
                    print(f"  [GRAD SKIP] batch {n_batches} — non-finite grad norm "
                          f"(total skips: {nan_skips})", flush=True)
                continue
            optimizer.step()

        scheduler.step()

        running_loss += loss.item()
        for k, v in loss_dict.items():
            running_components[k] = running_components.get(k, 0.0) + v

        # Accumulate phase weights for logging
        if 'phase_weights' in outputs:
            pw = outputs['phase_weights'].detach().mean(dim=0).cpu()  # (num_phases,)
            if phase_weight_sum is None:
                phase_weight_sum = pw.clone()
            else:
                phase_weight_sum += pw
            phase_weight_count += 1

        # Progress logging every log_every batches
        if n_batches % log_every == 0:
            elapsed_sofar = time.time() - t0
            avg_loss_sofar = running_loss / n_batches
            batches_per_sec = n_batches / elapsed_sofar
            move_acc_sofar = correct_moves / max(total_moves, 1)
            print(f"  [TRAIN] batch {n_batches:,}/{len(loader):,} "
                  f"({100*n_batches/len(loader):.1f}%) "
                  f"loss={avg_loss_sofar:.4f} "
                  f"acc={move_acc_sofar:.3f} "
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

    # Average phase weights
    if phase_weight_sum is not None and phase_weight_count > 0:
        avg_phase = phase_weight_sum / phase_weight_count
        avg_comp['avg_phase_opening'] = avg_phase[0].item()
        avg_comp['avg_phase_midgame'] = avg_phase[1].item()
        avg_comp['avg_phase_endgame'] = avg_phase[2].item() if len(avg_phase) > 2 else 0.0

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
    wdl_correct = 0
    wdl_total = 0
    time_abs_err_sum = 0.0
    time_total = 0
    # Phase weight accumulators
    phase_weight_sum = None
    phase_weight_count = 0

    for batch in loader:
        cp = batch['current_planes'].to(device, non_blocking=True)
        cp = cp.to(memory_format=torch.channels_last)
        pf = batch['possible_from_sq'].to(device, non_blocking=True)
        pt = batch['possible_to_sq'].to(device, non_blocking=True)
        pp = batch['possible_promo'].to(device, non_blocking=True)
        ps = batch['possible_scalars'].to(device, non_blocking=True)
        pm = batch['possible_mask'].to(device, non_blocking=True)
        tab = batch['tabular'].to(device, non_blocking=True)
        aidx = batch['actual_idx'].to(device, non_blocking=True)

        targets = {
            'move_idx':        aidx,
            'is_mistake':      batch['is_mistake'].to(device, non_blocking=True),
            'win_prob_before': batch['win_prob_before'].to(device, non_blocking=True),
            'time_spent_log':  batch['time_spent_log'].to(device, non_blocking=True),
        }

        with autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(cp, pf, pt, pp, ps, pm, tab, actual_idx=aidx)
            loss, loss_dict = criterion(outputs, targets)

        running_loss += loss.item()
        for k, v in loss_dict.items():
            running_components[k] = running_components.get(k, 0.0) + v
        n_batches += 1

        # Phase weights
        if 'phase_weights' in outputs:
            pw = outputs['phase_weights'].detach().mean(dim=0).cpu()
            if phase_weight_sum is None:
                phase_weight_sum = pw.clone()
            else:
                phase_weight_sum += pw
            phase_weight_count += 1

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

        # WDL before accuracy
        if 'win_prob_before' in outputs and 'win_prob_before' in targets:
            wdl_pred = outputs['win_prob_before'].argmax(dim=-1)
            wdl_true = targets['win_prob_before'].argmax(dim=-1)
            wdl_correct += (wdl_pred == wdl_true).sum().item()
            wdl_total += wdl_true.numel()

        # Time MAE
        if 'time_spent' in outputs:
            time_pred = outputs['time_spent'].squeeze(-1)
            time_true = targets['time_spent_log']
            time_abs_err_sum += torch.abs(time_pred - time_true).sum().item()
            time_total += time_true.numel()

    avg_loss = running_loss / max(n_batches, 1)
    avg_comp = {k: v / max(n_batches, 1) for k, v in running_components.items()}
    metrics = {
        'val_loss': avg_loss,
        'move_top1': correct_top1 / max(total_moves, 1),
        'move_top3': correct_top3 / max(total_moves, 1),
        'move_top5': correct_top5 / max(total_moves, 1),
        'mistake_acc': mistake_correct / max(mistake_total, 1),
        'wdl_before_acc': wdl_correct / max(wdl_total, 1),
        'time_mae': time_abs_err_sum / max(time_total, 1),
    }

    # Average phase weights
    if phase_weight_sum is not None and phase_weight_count > 0:
        avg_phase = phase_weight_sum / phase_weight_count
        avg_comp['avg_phase_opening'] = avg_phase[0].item()
        avg_comp['avg_phase_midgame'] = avg_phase[1].item()
        avg_comp['avg_phase_endgame'] = avg_phase[2].item() if len(avg_phase) > 2 else 0.0
        metrics['avg_phase_opening'] = avg_comp['avg_phase_opening']
        metrics['avg_phase_midgame'] = avg_comp['avg_phase_midgame']
        metrics['avg_phase_endgame'] = avg_comp['avg_phase_endgame']

    return avg_loss, metrics, avg_comp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train MIMO V5 model')
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
    # --- Model preset (use this instead of manually setting channels/blocks/hidden) ---
    parser.add_argument('--preset', type=str, default='v5',
                        choices=['v5-minimal', 'v5', 'v5-large', 'v5-widehead'],
                        help='Model preset: v5-minimal (~3.9M), v5 (~9.7M, default), '
                             'v5-large (~19.5M), v5-widehead (~10.55M, phase off + wide single heads). '
                             'Individual arch flags below override preset values.')
    # --- Model (override preset values when set explicitly) ---
    parser.add_argument('--max-possible', type=int, default=220)
    parser.add_argument('--cnn-channels', type=int, default=None,
                        help='CNN channels (override preset; v5-minimal=128, v5=192, v5-large=256)')
    parser.add_argument('--res-blocks', type=int, default=None,
                        help='Residual blocks (override preset; v5-minimal=6, v5=8, v5-large=10)')
    parser.add_argument('--hidden-dim', type=int, default=None,
                        help='Hidden dim (override preset; v5-minimal=256, v5=384, v5-large=512)')
    # --- Component versions (registry) ---
    parser.add_argument('--mistake-expert-ver', type=str, default=None,
                        help='Expert architecture for mistake head (override preset; see register_expert)')
    parser.add_argument('--time-expert-ver', type=str, default=None,
                        help='Expert architecture for time head (override preset)')
    parser.add_argument('--wdl-expert-ver', type=str, default=None,
                        help='Expert architecture for WDL head (override preset)')
    parser.add_argument('--move-head-ver', type=str, default=None,
                        help='Move head architecture (override preset; default=attention_deep)')
    # --- V5: Contrastive ---
    parser.add_argument('--contrastive-embed-dim', type=int, default=None,
                        help='Contrastive embedding dimension (0 to disable; override preset)')
    parser.add_argument('--contrastive-hidden-dim', type=int, default=None,
                        help='Hidden dim in contrastive encoder (override preset)')
    parser.add_argument('--contrastive-margin', type=float, default=1.0,
                        help='Triplet margin for contrastive loss')
    # --- V5: Phase-gated experts ---
    parser.add_argument('--use-phase-experts', dest='use_phase_experts', action='store_true', default=None,
                        help='Enable phase-gated experts (overrides preset)')
    parser.add_argument('--no-phase-experts', dest='use_phase_experts', action='store_false',
                        help='Disable phase-gated experts (V4-style single experts; overrides preset)')
    parser.add_argument('--phase-hidden-dim', type=int, default=None,
                        help='Hidden dim in PhaseEncoder (override preset)')
    parser.add_argument('--num-phases', type=int, default=3,
                        help='Number of game phases (default 3)')
    # --- FiLM conditioning ---
    parser.add_argument('--use-film', action='store_true', default=True,
                        help='Enable FiLM conditioning on Elo (default True)')
    parser.add_argument('--no-film', dest='use_film', action='store_false',
                        help='Disable FiLM conditioning')
    parser.add_argument('--film-hidden-dim', type=int, default=None,
                        help='FiLM conditioner hidden dim (override preset)')
    # --- Tactical enrichment ---
    parser.add_argument('--use-tactical-enrichment', action='store_true', default=False,
                        help='Enable frozen tactical/opening preprocessor trunk enrichment')
    parser.add_argument('--tactical-preprocessor-path', type=str, default=None,
                        help='Path to trained TacticalPreprocessor checkpoint (.pt)')
    # --- Data loading ---
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers (2 default on Windows 32GB; try 4 on 64GB+ or Linux)')
    parser.add_argument('--cache-shards', type=int, default=2,
                        help='Number of shards to keep in LRU cache per worker (mmap = cheap)')
    parser.add_argument('--with-phase', action='store_true', help='Include game phase feature')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='DataLoader prefetch factor per worker')
    parser.add_argument('--pin-memory', action='store_true', default=False,
                        help='Enable pin_memory for DataLoader (safe for V3 small tensors)')
    parser.add_argument('--no-npy-cache', action='store_true', default=False,
                        help='Load shards directly from .npz into RAM instead of extracting '
                             'to .npy cache on disk. Saves ~500-600 GB disk at cost of ~1-2s '
                             'decompression per shard load (overlapped with GPU via prefetch).')
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
    parser.add_argument('--log-every', type=int, default=500,
                        help='Print training progress every N batches (default 500)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g. checkpoints/latest.pt)')
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    np.random.seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        # Speedup #1: enable TF32 tensor cores for float32 matmul/conv on Ada.
        # Keeps FP32 range (incl. the float32 contrastive path), trades a few
        # mantissa bits in the matmul accumulate for a large throughput win.
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    use_amp = ('cuda' in args.device) and not args.no_amp
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    train_path, val_path = resolve_data_paths(args)
    print(f"\n[DATA] train: {train_path}")
    print(f"[DATA] val:   {val_path}")

    train_ds = MIMOCompactDataset(train_path, max_possible=args.max_possible,
                                  cache_shards=args.cache_shards, with_phase=args.with_phase,
                                  no_npy_cache=args.no_npy_cache)
    val_ds   = MIMOCompactDataset(val_path, max_possible=args.max_possible,
                                  cache_shards=args.cache_shards, with_phase=args.with_phase,
                                  no_npy_cache=args.no_npy_cache)

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
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    print(f"[DATA] train: {len(train_ds):,} examples → {len(train_loader):,} batches")
    print(f"[DATA] val:   {len(val_ds):,} examples → {len(val_loader):,} batches")

    # ---- Model (via preset + CLI overrides) ----
    # Build overrides dict from CLI args that were explicitly set (non-None).
    # These override the preset values; None means "use preset default".
    preset_overrides = {}
    _cli_to_preset = {
        'cnn_channels': 'cnn_channels',
        'res_blocks': 'num_res_blocks',
        'hidden_dim': 'hidden_dim',
        'move_head_ver': 'move_head_ver',
        'contrastive_embed_dim': 'contrastive_embed_dim',
        'contrastive_hidden_dim': 'contrastive_hidden_dim',
        'film_hidden_dim': 'film_hidden_dim',
        'phase_hidden_dim': 'phase_hidden_dim',
        'mistake_expert_ver': 'mistake_expert_ver',
        'time_expert_ver': 'time_expert_ver',
        'wdl_expert_ver': 'wdl_expert_ver',
        'use_phase_experts': 'use_phase_experts',
    }
    for cli_attr, preset_key in _cli_to_preset.items():
        val = getattr(args, cli_attr)
        if val is not None:
            preset_overrides[preset_key] = val

    # Non-preset args always passed directly
    preset_overrides.update({
        'tabular_dim': 20,
        'max_possible': args.max_possible,
        'move_scalar_dim': 13,
        'contrastive_margin': args.contrastive_margin,
        'num_phases': args.num_phases,
        'use_film': args.use_film,
        'use_tactical_enrichment': args.use_tactical_enrichment,
        'tactical_preprocessor_path': args.tactical_preprocessor_path,
    })

    print(f"\n[MODEL] Using preset: {args.preset}")
    model = ChessMIMOModelV5.from_preset(args.preset, **preset_overrides).to(device)
    # Speedup #2: channels_last layout — Ada conv kernels are faster on NHWC.
    # Pure memory-layout change; outputs are numerically identical.
    if 'cuda' in args.device:
        model = model.to(memory_format=torch.channels_last)

    n_params = sum(p.numel() for p in model.parameters())
    # Read effective config from the model (preset + overrides merged)
    _eff = ChessMIMOModelV5.PRESETS.get(args.preset, {})
    _eff.update(preset_overrides)
    print(f"\n[MODEL] ChessMIMOModelV5 (preset={args.preset}) — {n_params:,} parameters")
    print(f"        CNN channels={_eff.get('cnn_channels', '?')}, "
          f"res_blocks={_eff.get('num_res_blocks', '?')}, "
          f"hidden={_eff.get('hidden_dim', '?')}, max_moves={args.max_possible}")
    print(f"        Versions: {model.component_versions}")
    print(f"        Contrastive: embed_dim={_eff.get('contrastive_embed_dim', '?')}, "
          f"margin={args.contrastive_margin}")
    print(f"        Phase experts: {model.use_phase_experts} "
          f"(phases={args.num_phases}, hidden={_eff.get('phase_hidden_dim', '?')})")
    print(f"        FiLM: {args.use_film} (hidden={_eff.get('film_hidden_dim', '?')})")
    if args.use_tactical_enrichment:
        print(f"        Tactical enrichment: ON (path={args.tactical_preprocessor_path})")

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

    # ---- torch.compile (max-autotune for best Triton kernels, no CUDA graphs) ----
    if not args.no_compile and not args.use_ort and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='max-autotune-no-cudagraphs', fullgraph=True)
            print("[MODEL] torch.compile(mode='max-autotune-no-cudagraphs', fullgraph=True) enabled")
        except Exception as e:
            print(f"[MODEL] torch.compile failed, continuing without: {e}")

    # ---- Optimiser / scheduler ----
    criterion = MIMOLossV5(contrastive_margin=args.contrastive_margin).to(device)
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay, fused=True)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    # bfloat16 AMP does not need loss scaling — GradScaler disabled
    scaler = GradScaler('cuda', enabled=False)

    # ---- Resume from checkpoint ----
    start_epoch = 1
    resume_from_batch = 0
    best_val_loss = float('inf')
    history = []
    if args.resume:
        print(f"[RESUME] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        # Handle V4 → V5 checkpoint loading
        state_dict = ckpt['model_state_dict']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[V5] New modules with fresh weights ({len(missing)} keys):")
            v5_prefixes = ('contrastive_encoder.', 'contrastive_anchor_proj.',
                           'phase_encoder.', 'phase_experts.')
            new_module_names = set()
            for k in missing:
                prefix = k.split('.')[0]
                if prefix not in new_module_names:
                    new_module_names.add(prefix)
                    print(f"  → {prefix}.*")

        if 'criterion_state_dict' in ckpt:
            try:
                criterion.load_state_dict(ckpt['criterion_state_dict'], strict=False)
            except Exception as e:
                print(f"[RESUME] Criterion load partial (V4→V5 upgrade): {e}")

        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        if ckpt.get('mid_epoch'):
            start_epoch = ckpt['epoch']
            resume_from_batch = ckpt['batch']
            print(f"[RESUME] Mid-epoch resume: epoch {start_epoch}, batch {resume_from_batch:,}")
        else:
            start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        if 'scheduler_step' in ckpt:
            scheduler._step = ckpt['scheduler_step']
            for pg, base_lr in zip(scheduler.optimizer.param_groups, scheduler.base_lrs):
                if scheduler._step <= scheduler.warmup_steps:
                    scale = scheduler._step / max(scheduler.warmup_steps, 1)
                else:
                    progress = (scheduler._step - scheduler.warmup_steps) / max(
                        scheduler.total_steps - scheduler.warmup_steps, 1)
                    scale = 0.5 * (1.0 + math.cos(math.pi * progress))
                pg['lr'] = max(scheduler.min_lr, base_lr * scale)
        else:
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
    config['component_versions'] = model.component_versions
    config['n_params'] = n_params
    config['train_examples'] = len(train_ds)
    config['val_examples'] = len(val_ds)
    # Merge effective preset values into config so checkpoint is self-describing
    _preset_cfg = dict(ChessMIMOModelV5.PRESETS.get(args.preset, {}))
    _preset_cfg.update(preset_overrides)
    config['effective_preset'] = args.preset
    config['effective_config'] = _preset_cfg
    config['contrastive_margin'] = args.contrastive_margin
    config['use_phase_experts'] = model.use_phase_experts
    config['num_phases'] = args.num_phases
    config['tabular_dim'] = 20
    config['move_scalar_dim'] = 13
    config['use_film'] = args.use_film
    config['use_tactical_enrichment'] = args.use_tactical_enrichment
    if hasattr(model, '_tactical_config') and model._tactical_config is not None:
        config['tactical_preprocessor_config'] = model._tactical_config
    config['film_hidden_dim'] = args.film_hidden_dim
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
            resume_from_batch=resume_from_batch if epoch == start_epoch else 0,
            log_every=args.log_every)
        # Clear mid-epoch resume after first epoch processes
        if epoch == start_epoch:
            resume_from_batch = 0

        val_loss, val_metrics, val_comp = validate(
            model, val_loader, criterion, device, use_amp)

        lr_now = scheduler.lr
        print(f"\nEpoch {epoch}/{args.epochs}  ({elapsed:.0f}s)  lr={lr_now:.2e}")
        # Train summary with per-head losses
        train_loss_str = f"  TRAIN loss={train_loss:.4f}  move_acc={train_acc:.3f}"
        for head in ['move_logits', 'mistake_prob', 'win_prob_before', 'time_spent', 'contrastive']:
            if head in train_comp:
                train_loss_str += f"  {head}={train_comp[head]:.4f}"
        print(train_loss_str)

        # Phase weights
        if 'avg_phase_opening' in train_comp:
            print(f"  PHASE  opening={train_comp['avg_phase_opening']:.3f}  "
                  f"midgame={train_comp['avg_phase_midgame']:.3f}  "
                  f"endgame={train_comp['avg_phase_endgame']:.3f}")

        # Val summary with per-head losses and accuracies
        print(f"  VAL   loss={val_loss:.4f}  top1={val_metrics['move_top1']:.3f}  "
              f"top3={val_metrics['move_top3']:.3f}  top5={val_metrics['move_top5']:.3f}  "
              f"mistake_acc={val_metrics['mistake_acc']:.3f}  "
              f"wdl_acc={val_metrics.get('wdl_before_acc',0):.3f}  "
              f"time_mae={val_metrics.get('time_mae',0):.3f}")

        # Per-head val losses
        val_loss_str = '  LOSSES'
        for head in ['move_logits', 'mistake_prob', 'win_prob_before', 'time_spent', 'contrastive']:
            if head in val_comp:
                val_loss_str += f"  {head}={val_comp[head]:.4f}"
        print(val_loss_str)

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
            writer.add_scalar('Acc/wdl_before', val_metrics.get('wdl_before_acc',0), epoch)
            writer.add_scalar('Reg/time_mae', val_metrics.get('time_mae',0), epoch)
            writer.add_scalar('LR', lr_now, epoch)
            for k, v in val_comp.items():
                if k.startswith('w_'):
                    writer.add_scalar(f'Weights/{k}', v, epoch)
            for head in ['move_logits', 'mistake_prob', 'win_prob_before', 'time_spent', 'contrastive']:
                if head in val_comp:
                    writer.add_scalar(f'Loss/val_{head}', val_comp[head], epoch)
                if f'train_{head}' in entry:
                    writer.add_scalar(f'Loss/train_{head}', entry[f'train_{head}'], epoch)
            # Phase weights
            if 'avg_phase_opening' in val_comp:
                writer.add_scalar('Phase/opening', val_comp['avg_phase_opening'], epoch)
                writer.add_scalar('Phase/midgame', val_comp['avg_phase_midgame'], epoch)
                writer.add_scalar('Phase/endgame', val_comp['avg_phase_endgame'], epoch)

        # Checkpointing
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
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
