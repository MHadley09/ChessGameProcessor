#!/usr/bin/env python3
"""
validate_mimo.py — Comprehensive validation for all 5 MIMO heads.

Metrics computed:
    1. move_logits:       Top-1/3/5 accuracy, perplexity
    2. mistake_prob:      AUC-ROC, AUC-PR, calibration
    3. win_prob_before:   Brier score, ECE, accuracy (with leakage check)
    4. win_prob_after:    Brier score, ECE, accuracy
    5. time_spent:        MAE, RMSE, Pearson r, Spearman ρ, bucket calibration
    6. Leakage detector:  before should be LESS accurate than after

Outputs:
    - metrics.json       — all numbers
    - report.txt         — human-readable summary
    - calibration.png    — calibration curves (4 subplots)

Usage:
    python validate_mimo.py \
        --checkpoint checkpoints/best.pt \
        --data       data/mimo_dataset/test.npz \
        --output-dir validation_results
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from chess_mimo_model import ChessMIMOModel
from mimo_dataset import MIMOCompactDataset

# Optional sklearn / matplotlib
try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        brier_score_loss, mean_absolute_error, mean_squared_error,
    )
    from scipy.stats import pearsonr, spearmanr
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn/scipy not found — some metrics will be skipped")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Dataset — uses MIMOCompactDataset from mimo_dataset.py
# (planes built on-the-fly from FEN, no pre-stored planes)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Collector: run model and gather predictions
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    preds = defaultdict(list)
    targs = defaultdict(list)
    meta  = defaultdict(list)

    for batch in loader:
        cp = batch['current_planes'].to(device)
        pp = batch['possible_planes'].to(device)
        ps = batch['possible_scalars'].to(device)
        pm = batch['possible_mask'].to(device)
        tab = batch['tabular'].to(device)
        aidx = batch['actual_idx'].to(device)
        gphase = batch['game_phase'].to(device)

        outputs = model(cp, pp, ps, pm, tab, actual_idx=aidx, game_phase=gphase)

        preds['move_logits'].append(
            torch.softmax(outputs['move_logits'], dim=-1).cpu().numpy())
        preds['mistake_prob'].append(outputs['mistake_prob'].cpu().numpy())
        if 'win_prob_before' in outputs:
            preds['win_prob_before'].append(outputs['win_prob_before'].cpu().numpy())
        if 'win_prob_after' in outputs:
            preds['win_prob_after'].append(outputs['win_prob_after'].cpu().numpy())
        preds['time_spent'].append(outputs['time_spent'].cpu().numpy())

        targs['move_idx'].append(batch['actual_idx'].numpy())
        targs['is_mistake'].append(batch['is_mistake'].numpy())
        targs['win_prob_before'].append(batch['win_prob_before'].numpy())
        targs['win_prob_after'].append(batch['win_prob_after'].numpy())
        targs['time_spent_log'].append(batch['time_spent_log'].numpy())

        meta['move_no'].append(batch['tabular'][:, 4].numpy() * 200)  # un-normalise
        meta['elo_avg'].append(
            ((batch['tabular'][:, 1] + batch['tabular'][:, 2]) * 3000 / 2).numpy())
        meta['game_phase'].append(batch['game_phase'].numpy())

    for k in preds:
        preds[k] = np.concatenate(preds[k], axis=0)
    for k in targs:
        targs[k] = np.concatenate(targs[k], axis=0)
    for k in meta:
        meta[k] = np.concatenate(meta[k], axis=0)

    return preds, targs, meta


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def compute_all_metrics(preds, targs, meta):
    metrics = {}
    N = len(targs['move_idx'])
    print(f"\nValidating on {N:,} examples\n{'='*60}")

    # ------------------------------------------------------------------
    # 1. Move prediction
    # ------------------------------------------------------------------
    print("\n1. MOVE PREDICTION")
    print('-' * 40)
    move_probs = preds['move_logits']
    move_targ = targs['move_idx']

    top1 = top3 = top5 = 0
    log_probs = []
    for i in range(N):
        ranked = np.argsort(move_probs[i])[::-1]
        t = move_targ[i]
        if ranked[0] == t: top1 += 1
        if t in ranked[:3]: top3 += 1
        if t in ranked[:5]: top5 += 1
        log_probs.append(np.log(max(move_probs[i, t], 1e-10)))

    metrics['move_top1'] = top1 / N
    metrics['move_top3'] = top3 / N
    metrics['move_top5'] = top5 / N
    metrics['move_perplexity'] = float(np.exp(-np.mean(log_probs)))
    for k in ['move_top1', 'move_top3', 'move_top5', 'move_perplexity']:
        print(f"  {k:25s} {metrics[k]:.4f}")

    # By game phase (move_no heuristic)
    move_nos = meta['move_no']
    for phase, lo, hi in [('opening', 0, 15), ('middlegame', 15, 40), ('endgame', 40, 999)]:
        mask = (move_nos >= lo) & (move_nos < hi)
        if mask.sum() > 10:
            phase_top1 = sum(
                np.argsort(move_probs[i])[-1] == move_targ[i]
                for i in np.where(mask)[0]) / mask.sum()
            metrics[f'move_top1_{phase}'] = float(phase_top1)
            print(f"  move_top1_{phase:12s}  {phase_top1:.4f}  (n={mask.sum()})")

    # By game phase (embedding-based classification from FEN)
    if 'game_phase' in meta:
        gp = meta['game_phase']
        for phase_idx, phase_name in [(0, 'opening_emb'), (1, 'middlegame_emb'), (2, 'endgame_emb')]:
            mask = (gp == phase_idx)
            if mask.sum() > 10:
                phase_top1 = sum(
                    np.argsort(move_probs[i])[-1] == move_targ[i]
                    for i in np.where(mask)[0]) / mask.sum()
                metrics[f'move_top1_{phase_name}'] = float(phase_top1)
                print(f"  move_top1_{phase_name:12s}  {phase_top1:.4f}  (n={mask.sum()})")

    # ------------------------------------------------------------------
    # 2. Mistake prediction
    # ------------------------------------------------------------------
    print("\n2. MISTAKE PREDICTION")
    print('-' * 40)
    mis_pred = preds['mistake_prob'].flatten()
    mis_targ = targs['is_mistake'].flatten()
    valid = ~np.isnan(mis_targ)
    if valid.sum() > 0 and HAS_SKLEARN:
        mp, mt = mis_pred[valid], mis_targ[valid]
        if len(np.unique(mt)) > 1:
            metrics['mistake_auc_roc'] = float(roc_auc_score(mt, mp))
            metrics['mistake_auc_pr']  = float(average_precision_score(mt, mp))
        metrics['mistake_pos_rate'] = float(mt.mean())
        metrics['mistake_acc']      = float(((mp > 0.5) == mt).mean())
        for k in ['mistake_auc_roc', 'mistake_auc_pr', 'mistake_pos_rate', 'mistake_acc']:
            if k in metrics:
                print(f"  {k:25s} {metrics[k]:.4f}")

    # By Elo bucket
    elo_avg = meta['elo_avg']
    for elo_name, lo, hi in [('<1500', 0, 1500), ('1500-2000', 1500, 2000), ('2000+', 2000, 9999)]:
        mask = valid & (elo_avg >= lo) & (elo_avg < hi)
        if mask.sum() > 10 and len(np.unique(mis_targ[mask])) > 1 and HAS_SKLEARN:
            auc = roc_auc_score(mis_targ[mask], mis_pred[mask])
            metrics[f'mistake_auc_{elo_name}'] = float(auc)
            print(f"  mistake_auc_{elo_name:10s}  {auc:.4f}  (n={mask.sum()})")

    # ------------------------------------------------------------------
    # 3 & 4. Win probability before / after
    # ------------------------------------------------------------------
    for key in ['win_prob_before', 'win_prob_after']:
        if key not in preds:
            continue
        print(f"\n3/4. {key.upper()}")
        print('-' * 40)
        p = preds[key]
        t = targs[key]

        pred_cls = np.argmax(p, axis=1)
        true_cls = np.argmax(t, axis=1)
        acc = float((pred_cls == true_cls).mean())
        metrics[f'{key}_accuracy'] = acc

        if HAS_SKLEARN:
            brier = float(np.mean([brier_score_loss(t[:, i], p[:, i]) for i in range(3)]))
            metrics[f'{key}_brier'] = brier
        else:
            brier = None

        # ECE
        confs = np.max(p, axis=1)
        accs  = (pred_cls == true_cls).astype(float)
        ece = _ece(confs, accs)
        metrics[f'{key}_ece'] = float(ece)

        print(f"  accuracy  {acc:.4f}")
        if brier is not None:
            print(f"  brier     {brier:.4f}")
        print(f"  ECE       {ece:.4f}")
        for i, name in enumerate(['win', 'draw', 'loss']):
            print(f"    {name}: pred_mean={p[:, i].mean():.3f}  true_mean={t[:, i].mean():.3f}")

    # ------------------------------------------------------------------
    # 5. Time spent
    # ------------------------------------------------------------------
    print("\n5. TIME SPENT")
    print('-' * 40)
    tp_log = preds['time_spent'].flatten()
    tt_log = targs['time_spent_log'].flatten()
    tp = np.expm1(tp_log)
    tt = np.expm1(tt_log)
    valid = (tt < 600) & (tt >= 0)
    if valid.sum() > 10 and HAS_SKLEARN:
        tpv, ttv = tp[valid], tt[valid]
        metrics['time_mae']     = float(mean_absolute_error(ttv, tpv))
        metrics['time_rmse']    = float(np.sqrt(mean_squared_error(ttv, tpv)))
        metrics['time_pearson'] = float(pearsonr(ttv, tpv)[0])
        metrics['time_spearman']= float(spearmanr(ttv, tpv)[0])
        for k in ['time_mae', 'time_rmse', 'time_pearson', 'time_spearman']:
            print(f"  {k:25s} {metrics[k]:.4f}")

        # Bucket calibration
        for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 30), (30, 600)]:
            bmask = (ttv >= lo) & (ttv < hi)
            if bmask.sum() > 10:
                bmae = float(np.mean(np.abs(tpv[bmask] - ttv[bmask])))
                print(f"    [{lo:3d}, {hi:3d})s  n={bmask.sum():5d}  MAE={bmae:.2f}s")

    # ------------------------------------------------------------------
    # 6. Leakage check
    # ------------------------------------------------------------------
    print(f"\n6. LEAKAGE CHECK")
    print('-' * 40)
    before_acc = metrics.get('win_prob_before_accuracy', 0)
    after_acc  = metrics.get('win_prob_after_accuracy', 0)
    print(f"  win_prob_before accuracy: {before_acc:.4f}")
    print(f"  win_prob_after  accuracy: {after_acc:.4f}")

    if before_acc > 0.85:
        print("  ⚠  WARNING: win_prob_before suspiciously high — possible masking failure!")
        metrics['leakage_warning'] = True
    elif before_acc > after_acc + 0.05:
        print("  ⚠  WARNING: before MORE accurate than after — masking is likely broken!")
        metrics['leakage_warning'] = True
    else:
        gap = after_acc - before_acc
        print(f"  ✓  Masking looks correct (after is {gap:.3f} more accurate)")
        metrics['leakage_warning'] = False

    print('=' * 60)
    return metrics


def _ece(confidences, accuracies, n_bins=10):
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_bounds[i]) & (confidences <= bin_bounds[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() / len(confidences) * abs(
                accuracies[mask].mean() - confidences[mask].mean())
    return ece


# ---------------------------------------------------------------------------
# Calibration plots
# ---------------------------------------------------------------------------

def plot_calibration(preds, targs, output_dir):
    if not HAS_MPL:
        print("[SKIP] matplotlib not available — skipping calibration plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Mistake
    ax = axes[0, 0]
    mp = preds['mistake_prob'].flatten()
    mt = targs['is_mistake'].flatten()
    _plot_binary_cal(ax, mp[~np.isnan(mt)], mt[~np.isnan(mt)], 'Mistake Prob')

    # WDL before
    if 'win_prob_before' in preds:
        _plot_multiclass_cal(axes[0, 1], preds['win_prob_before'],
                             targs['win_prob_before'], 'WDL Before')

    # WDL after
    if 'win_prob_after' in preds:
        _plot_multiclass_cal(axes[1, 0], preds['win_prob_after'],
                             targs['win_prob_after'], 'WDL After')

    # Time
    ax = axes[1, 1]
    tp = np.expm1(preds['time_spent'].flatten())
    tt = np.expm1(targs['time_spent_log'].flatten())
    valid = (tt < 120) & (tt >= 0)
    if valid.sum() > 0:
        ax.scatter(tt[valid], tp[valid], alpha=0.05, s=1)
        mx = min(60, tt[valid].max())
        ax.plot([0, mx], [0, mx], 'r--')
        ax.set(xlabel='Actual (s)', ylabel='Predicted (s)', title='Time Spent',
               xlim=(0, mx), ylim=(0, mx))

    plt.tight_layout()
    path = Path(output_dir) / 'calibration.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved calibration plot → {path}")


def _plot_binary_cal(ax, preds, targets, title, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs = [], []
    for i in range(n_bins):
        mask = (preds > bins[i]) & (preds <= bins[i + 1])
        if mask.sum() > 0:
            centers.append((bins[i] + bins[i + 1]) / 2)
            accs.append(targets[mask].mean())
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.plot(centers, accs, 'o-')
    ax.set(xlabel='Predicted', ylabel='Observed', title=title)
    ax.grid(True, alpha=0.3)


def _plot_multiclass_cal(ax, preds, targets, title, n_bins=10):
    confs = np.max(preds, axis=1)
    correct = (np.argmax(preds, axis=1) == np.argmax(targets, axis=1)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs = [], []
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i + 1])
        if mask.sum() > 0:
            centers.append((bins[i] + bins[i + 1]) / 2)
            accs.append(correct[mask].mean())
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.plot(centers, accs, 'o-')
    ax.set(xlabel='Confidence', ylabel='Accuracy', title=title)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Validate MIMO')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data',       required=True, help='.npz test set')
    parser.add_argument('--output-dir', default='validation_results')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})

    model = ChessMIMOModel(
        cnn_channels=cfg.get('cnn_channels', 128),
        num_res_blocks=cfg.get('res_blocks', 6),
        tabular_dim=18,
        max_possible=cfg.get('max_possible', 40),
        hidden_dim=cfg.get('hidden_dim', 256),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded epoch {ckpt.get('epoch', '?')} — "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    # Load data
    ds = MIMOCompactDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Validating on {len(ds):,} examples …")

    # Run
    preds, targs, meta = collect_predictions(model, loader, device)
    metrics = compute_all_metrics(preds, targs, meta)
    plot_calibration(preds, targs, str(out))

    # Save
    with open(out / 'metrics.json', 'w') as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                   for k, v in metrics.items()}, f, indent=2)
    with open(out / 'report.txt', 'w') as f:
        f.write("MIMO OPUS VALIDATION REPORT\n" + "=" * 60 + "\n\n")
        for k, v in sorted(metrics.items()):
            f.write(f"{k:35s}  {v}\n")
    print(f"\n✓ Saved metrics.json + report.txt → {out}")


if __name__ == '__main__':
    main()
