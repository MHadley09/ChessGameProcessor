#!/usr/bin/env python3
"""
validate_mimo.py — Comprehensive validation for all MIMO V5 heads.

Metrics computed:
    1. move_logits:       Top-1/3/5 accuracy, perplexity
    2. mistake_prob:      AUC-ROC, AUC-PR, calibration
    3. win_prob_before:   Brier score, ECE, accuracy
    4. time_spent:        MAE, RMSE, Pearson r, Spearman ρ, bucket calibration
    5. contrastive:       Triplet accuracy, mean d_pos, mean d_neg, margin satisfaction rate
    6. phase_weights:     Phase distribution, per-phase-bucket accuracy

Outputs:
    - metrics.json       — all numbers
    - report.txt         — human-readable summary
    - calibration.png    — calibration curves

Usage (sharded — preferred):
    python validate_mimo.py \
        --checkpoint checkpoints/best.pt \
        --data-dir   dataset/opus \
        --split      test \
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

from chess_mimo_model_v5 import ChessMIMOModelV5, MIMOLossV5
from mimo_dataset_polars import MIMOCompactDataset

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
# Resolve data path
# ---------------------------------------------------------------------------

def resolve_data_path(args):
    """Return the data path — either a shard directory or a single .npz file."""
    if args.data_dir:
        data_dir = Path(args.data_dir)
        split_path = data_dir / args.split
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"No {args.split}/ subdirectory in {data_dir}. "
                f"Available: {[d.name for d in data_dir.iterdir() if d.is_dir()]}"
            )
        return str(split_path)
    elif args.data:
        return args.data
    else:
        raise ValueError("Provide --data-dir (sharded) or --data (legacy single .npz)")


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
        pf = batch['possible_from_sq'].to(device)
        pt = batch['possible_to_sq'].to(device)
        pp = batch['possible_promo'].to(device)
        ps = batch['possible_scalars'].to(device)
        pm = batch['possible_mask'].to(device)
        tab = batch['tabular'].to(device)
        aidx = batch['actual_idx'].to(device)

        outputs = model(cp, pf, pt, pp, ps, pm, tab, actual_idx=aidx)

        preds['move_logits'].append(
            torch.softmax(outputs['move_logits'], dim=-1).cpu().numpy())
        preds['mistake_prob'].append(outputs['mistake_prob'].sigmoid().cpu().numpy())
        if 'win_prob_before' in outputs:
            preds['win_prob_before'].append(outputs['win_prob_before'].cpu().numpy())
        preds['time_spent'].append(outputs['time_spent'].cpu().numpy())

        # Contrastive outputs
        if 'contrastive_embed' in outputs:
            preds['contrastive_embed'].append(outputs['contrastive_embed'].cpu().numpy())
            preds['contrastive_anchor'].append(outputs['contrastive_anchor'].cpu().numpy())

        # Phase weights
        if 'phase_weights' in outputs:
            preds['phase_weights'].append(outputs['phase_weights'].cpu().numpy())

        targs['move_idx'].append(batch['actual_idx'].numpy())
        targs['is_mistake'].append(batch['is_mistake'].numpy())
        targs['win_prob_before'].append(batch['win_prob_before'].numpy())
        targs['time_spent_log'].append(batch['time_spent_log'].numpy())

        meta['move_no'].append(batch['tabular'][:, 4].numpy() * 200)  # un-normalise
        meta['elo_avg'].append(
            ((batch['tabular'][:, 1] + batch['tabular'][:, 2]) * 3000 / 2).numpy())

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

    # By game phase
    move_nos = meta['move_no']
    for phase, lo, hi in [('opening', 0, 15), ('middlegame', 15, 40), ('endgame', 40, 999)]:
        mask = (move_nos >= lo) & (move_nos < hi)
        if mask.sum() > 10:
            phase_top1 = sum(
                np.argsort(move_probs[i])[-1] == move_targ[i]
                for i in np.where(mask)[0]) / mask.sum()
            metrics[f'move_top1_{phase}'] = float(phase_top1)
            print(f"  move_top1_{phase:12s}  {phase_top1:.4f}  (n={mask.sum()})")

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
    for key in ['win_prob_before']:
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
    # 6. Contrastive metrics (V5)
    # ------------------------------------------------------------------
    if 'contrastive_embed' in preds and 'contrastive_anchor' in preds:
        print("\n6. CONTRASTIVE METRICS")
        print('-' * 40)
        c_embed = preds['contrastive_embed']    # (N, M, 64)
        c_anchor = preds['contrastive_anchor']  # (N, 64)
        actual = targs['move_idx']              # (N,)
        move_probs_c = preds['move_logits']     # (N, M)

        valid_c = actual >= 0
        if valid_c.sum() > 0:
            va = c_anchor[valid_c]
            ve = c_embed[valid_c]
            vact = actual[valid_c]
            vmp = move_probs_c[valid_c]
            Vc = va.shape[0]

            # Positive: embedding of chosen move
            pos = ve[np.arange(Vc), vact]  # (Vc, 64)

            # Near-miss: highest model score that isn't ground truth
            logits_nm = vmp.copy()
            logits_nm[np.arange(Vc), vact] = -np.inf
            nm_idx = logits_nm.argmax(axis=1)
            neg = ve[np.arange(Vc), nm_idx]  # (Vc, 64)

            d_pos = np.sqrt(np.sum((va - pos) ** 2, axis=1))
            d_neg = np.sqrt(np.sum((va - neg) ** 2, axis=1))

            metrics['contrastive_d_pos_mean'] = float(d_pos.mean())
            metrics['contrastive_d_neg_mean'] = float(d_neg.mean())
            metrics['contrastive_triplet_acc'] = float((d_pos < d_neg).mean())
            metrics['contrastive_margin_sat_1.0'] = float((d_neg - d_pos > 1.0).mean())

            for k in ['contrastive_d_pos_mean', 'contrastive_d_neg_mean',
                       'contrastive_triplet_acc', 'contrastive_margin_sat_1.0']:
                print(f"  {k:35s} {metrics[k]:.4f}")

    # ------------------------------------------------------------------
    # 7. Phase weights distribution (V5)
    # ------------------------------------------------------------------
    if 'phase_weights' in preds:
        print("\n7. PHASE WEIGHTS")
        print('-' * 40)
        pw = preds['phase_weights']  # (N, num_phases)
        phase_names = ['opening', 'midgame', 'endgame']
        for i, name in enumerate(phase_names[:pw.shape[1]]):
            metrics[f'phase_weight_{name}_mean'] = float(pw[:, i].mean())
            metrics[f'phase_weight_{name}_std'] = float(pw[:, i].std())
            print(f"  {name:12s}  mean={pw[:, i].mean():.3f}  std={pw[:, i].std():.3f}")

        # Phase weight vs move number (verify phase detection makes sense)
        move_nos = meta['move_no']
        for phase, lo, hi in [('opening', 0, 15), ('middlegame', 15, 40), ('endgame', 40, 999)]:
            mask = (move_nos >= lo) & (move_nos < hi)
            if mask.sum() > 10:
                pw_phase = pw[mask].mean(axis=0)
                print(f"  Moves [{lo:3d},{hi:3d}): "
                      f"open={pw_phase[0]:.3f} mid={pw_phase[1]:.3f} end={pw_phase[2]:.3f}"
                      f"  (n={mask.sum()})")

    # ------------------------------------------------------------------
    # 8. Leakage check
    # ------------------------------------------------------------------
    print(f"\n8. LEAKAGE CHECK")
    print('-' * 40)
    before_acc = metrics.get('win_prob_before_accuracy', 0)
    print(f"  win_prob_before accuracy: {before_acc:.4f}")

    if before_acc > 0.85:
        print("  ⚠  WARNING: win_prob_before suspiciously high — possible masking failure!")
        metrics['leakage_warning'] = True
    else:
        print(f"  ✓  win_prob_before accuracy looks reasonable")
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

    # Time
    ax = axes[1, 0]
    tp = np.expm1(preds['time_spent'].flatten())
    tt = np.expm1(targs['time_spent_log'].flatten())
    valid = (tt < 120) & (tt >= 0)
    if valid.sum() > 0:
        ax.scatter(tt[valid], tp[valid], alpha=0.05, s=1)
        mx = min(60, tt[valid].max())
        ax.plot([0, mx], [0, mx], 'r--')
        ax.set(xlabel='Actual (s)', ylabel='Predicted (s)', title='Time Spent',
               xlim=(0, mx), ylim=(0, mx))

    # Phase weights by move number
    ax = axes[1, 1]
    if 'phase_weights' in preds:
        pw = preds['phase_weights']
        move_nos = meta if isinstance(meta, dict) else {}
        # Simple bar chart of mean phase weights
        phase_names = ['Opening', 'Middlegame', 'Endgame'][:pw.shape[1]]
        means = [pw[:, i].mean() for i in range(pw.shape[1])]
        ax.bar(phase_names, means, color=['#4CAF50', '#2196F3', '#FF5722'])
        ax.set(ylabel='Mean Weight', title='Phase Weight Distribution')
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'No phase weights\n(phase experts disabled)',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Phase Weights')

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
    parser = argparse.ArgumentParser(description='Validate MIMO V5')
    parser.add_argument('--checkpoint', required=True)
    # --- Data source (pick one) ---
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Root dataset dir with train/val/test shard subdirs')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to validate on (used with --data-dir)')
    parser.add_argument('--data', type=str, default=None,
                        help='Legacy: single .npz or shard directory')
    # --- Data loading ---
    parser.add_argument('--cache-shards', type=int, default=2,
                        help='Number of shards to keep in LRU cache per worker (mmap = cheap)')
    parser.add_argument('--with-phase', action='store_true', help='Include game phase feature')
    # --- Misc ---
    parser.add_argument('--output-dir', default='validation_results')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers (reduced for Windows shared memory)')
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    comp_ver = cfg.get('component_versions', {})

    model = ChessMIMOModelV5(
        cnn_channels=cfg.get('cnn_channels', 128),
        num_res_blocks=cfg.get('res_blocks', 6),
        tabular_dim=cfg.get('tabular_dim', 20),
        max_possible=cfg.get('max_possible', 220),
        hidden_dim=cfg.get('hidden_dim', 256),
        move_scalar_dim=cfg.get('move_scalar_dim', 13),
        mistake_expert_ver=comp_ver.get('mistake_expert', 'default'),
        time_expert_ver=comp_ver.get('time_expert', 'default'),
        wdl_expert_ver=comp_ver.get('wdl_expert', 'default'),
        move_head_ver=comp_ver.get('move_head', 'default'),
        contrastive_embed_dim=cfg.get('contrastive_embed_dim', 64),
        contrastive_hidden_dim=cfg.get('contrastive_hidden_dim', 128),
        contrastive_margin=cfg.get('contrastive_margin', 1.0),
        use_phase_experts=cfg.get('use_phase_experts', True),
        phase_hidden_dim=cfg.get('phase_hidden_dim', 64),
        num_phases=cfg.get('num_phases', 3),
        use_film=cfg.get('use_film', True),
        film_hidden_dim=cfg.get('film_hidden_dim', 64),
        use_tactical_enrichment=cfg.get('use_tactical_enrichment', False),
        tactical_preprocessor_config=cfg.get('tactical_preprocessor_config'),
    ).to(device)

    # Handle V4→V5 or partial V5 checkpoint loading
    state_dict = ckpt['model_state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[V5] {len(missing)} missing keys (new V5 modules with fresh weights)")
    model.eval()
    print(f"Loaded epoch {ckpt.get('epoch', '?')} — "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    print(f"V5 features: contrastive_dim={model.contrastive_embed_dim}, "
          f"phase_experts={model.use_phase_experts}")

    # Load data
    data_path = resolve_data_path(args)
    print(f"[DATA] Loading: {data_path}")
    ds = MIMOCompactDataset(data_path, cache_shards=args.cache_shards,
                            with_phase=args.with_phase)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False,
                        prefetch_factor=2 if args.num_workers > 0 else None,
                        persistent_workers=args.num_workers > 0)
    print(f"[DATA] {len(ds):,} examples → {len(loader):,} batches")

    # Run
    preds, targs, meta = collect_predictions(model, loader, device)
    metrics = compute_all_metrics(preds, targs, meta)
    plot_calibration(preds, targs, str(out))

    # Save
    with open(out / 'metrics.json', 'w') as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                   for k, v in metrics.items()}, f, indent=2)
    with open(out / 'report.txt', 'w') as f:
        f.write("MIMO V5 VALIDATION REPORT\n" + "=" * 60 + "\n\n")
        for k, v in sorted(metrics.items()):
            f.write(f"{k:35s}  {v}\n")
    print(f"\n✓ Saved metrics.json + report.txt → {out}")


if __name__ == '__main__':
    main()
