#!/usr/bin/env python3
"""
train_mimo_opus.py — Training script for the MIMO Opus chess model.

Features:
    - Mixed-precision (AMP) training for RTX 4090 efficiency
    - Cosine annealing with linear warmup
    - Learnable uncertainty-based multi-task loss weighting
    - Gradient clipping
    - Single forward pass with built-in masking (no double forward pass)
    - TensorBoard logging
    - Proper checkpoint saving (best + latest + periodic)
    - Reproducible with --seed

Usage:
    python train_mimo_opus.py \
        --train-data data/mimo_opus_dataset/train.npz \
        --val-data   data/mimo_opus_dataset/val.npz \
        --output-dir checkpoints/opus_run1 \
        --epochs 30 \
        --batch-size 64 \
        --lr 3e-4 \
        --device cuda
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
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

from chess_mimo_model_opus import ChessMIMOModelOpus, MIMOLossOpus

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MIMONpzDataset(Dataset):
    """Loads a .npz file produced by mimo_dataset_opus.py."""

    def __init__(self, npz_path: str):
        print(f"[DATA] Loading {npz_path} …")
        data = np.load(npz_path)
        self.current_planes  = data['current_planes']     # (N, 47, 8, 8)
        self.possible_planes = data['possible_planes']     # (N, M, 47, 8, 8)
        self.possible_scalars= data['possible_scalars']    # (N, M, 6)
        self.possible_mask   = data['possible_mask']       # (N, M)
        self.tabular         = data['tabular']             # (N, 10)
        self.actual_idx      = data['actual_idx']          # (N,)
        self.is_mistake      = data['is_mistake']          # (N,)
        self.win_prob_before = data['win_prob_before']     # (N, 3)
        self.win_prob_after  = data['win_prob_after']      # (N, 3)
        self.time_spent_log  = data['time_spent_log']      # (N,)
        self.n = len(self.current_planes)
        print(f"[DATA] {self.n:,} examples loaded")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            'current_planes':  torch.from_numpy(self.current_planes[idx]).float(),
            'possible_planes': torch.from_numpy(self.possible_planes[idx]).float(),
            'possible_scalars':torch.from_numpy(self.possible_scalars[idx]).float(),
            'possible_mask':   torch.from_numpy(self.possible_mask[idx]).float(),
            'tabular':         torch.from_numpy(self.tabular[idx]).float(),
            'actual_idx':      torch.tensor(self.actual_idx[idx], dtype=torch.long),
            'is_mistake':      torch.tensor(self.is_mistake[idx], dtype=torch.float32),
            'win_prob_before': torch.from_numpy(self.win_prob_before[idx]).float(),
            'win_prob_after':  torch.from_numpy(self.win_prob_after[idx]).float(),
            'time_spent_log':  torch.tensor(self.time_spent_log[idx], dtype=torch.float32),
        }


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
# Train / validate one epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                device, epoch, use_amp):
    model.train()
    running_loss = 0.0
    running_components = {}
    n_batches = 0
    correct_moves = 0
    total_moves = 0
    t0 = time.time()

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

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(cp, pp, ps, pm, tab, actual_idx=aidx)
            loss, loss_dict = criterion(outputs, targets)

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
        n_batches += 1

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

        with autocast(enabled=use_amp):
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
        mistake_pred = (outputs['mistake_prob'].squeeze(-1) > 0.5).float()
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
    parser = argparse.ArgumentParser(description='Train MIMO Opus model')
    parser.add_argument('--train-data', required=True, help='.npz from mimo_dataset_opus')
    parser.add_argument('--val-data',   required=True, help='.npz for validation')
    parser.add_argument('--output-dir', default='checkpoints/opus', help='Output directory')
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch-size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-pct', type=float, default=0.05, help='Warmup fraction of total steps')
    parser.add_argument('--max-possible', type=int, default=40)
    parser.add_argument('--cnn-channels', type=int, default=128)
    parser.add_argument('--res-blocks', type=int, default=6)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-amp',     action='store_true', help='Disable mixed precision')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--save-every', type=int, default=1, help='Checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (e.g. checkpoints/opus/latest.pt)')
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    use_amp = ('cuda' in args.device) and not args.no_amp
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    train_ds = MIMONpzDataset(args.train_data)
    val_ds   = MIMONpzDataset(args.val_data)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Model ----
    model = ChessMIMOModelOpus(
        cnn_channels=args.cnn_channels,
        num_res_blocks=args.res_blocks,
        tabular_dim=18,
        max_possible=args.max_possible,
        hidden_dim=args.hidden_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[MODEL] ChessMIMOModelOpus — {n_params:,} parameters")
    print(f"        CNN channels={args.cnn_channels}, res_blocks={args.res_blocks}, "
          f"hidden={args.hidden_dim}, max_moves={args.max_possible}")

    # ---- Optimiser / scheduler ----
    criterion = MIMOLossOpus().to(device)
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=use_amp)

    # ---- Resume from checkpoint ----
    start_epoch = 1
    best_val_loss = float('inf')
    history = []
    if args.resume:
        print(f"[RESUME] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        criterion.load_state_dict(ckpt['criterion_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        # Fast-forward scheduler to correct step
        steps_done = ckpt['epoch'] * len(train_loader)
        for _ in range(steps_done):
            scheduler.step()
        print(f"[RESUME] Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    print(f"[TRAIN] epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, "
          f"warmup={warmup_steps} steps, total={total_steps} steps, AMP={use_amp}")

    # ---- Save config ----
    config = vars(args)
    config['n_params'] = n_params
    with open(out_dir / 'train_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ---- TensorBoard ----
    writer = None
    if HAS_TB:
        writer = SummaryWriter(log_dir=str(out_dir / 'tb'))

    # ---- Training loop ----

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc, train_comp, elapsed = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, epoch, use_amp)

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
