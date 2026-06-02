#!/usr/bin/env python3
"""
maia_vs_krut_eval.py — Move-prediction accuracy comparison: Maia-2 vs KRUT.

Maia-2 is a set of LC0 network weights run with nodes=1 (pure policy net,
no search).  KRUT is a MIMO V4 multi-head model.  This script compares their
move-prediction accuracy on a hold-out set, bucketed by the side-to-move Elo
in 100-point ranges.

Metrics (per bucket and overall):
  - Top-1 / Top-2 / Top-3 / Top-5 accuracy
  - Log-loss (cross-entropy): -log(P(actual_move)), averaged.
    Strongly penalises confident wrong predictions.
  - Move count (n)

Bucketing:
  Each position is assigned to the 100-Elo bucket of the *side to move*,
  e.g. a 1523-rated player's moves go into the 1500-1599 bucket regardless
  of the opponent's rating.

Data flow
─────────
  KRUT  →  NPZ shard dataset (MIMOCompactDataset), batch forward pass.
  Maia  →  LC0 subprocess (UCI) with Maia-2 weights, nodes=1,
           VerboseMoveStats=true.  FENs + actual moves come from the
           original parquet files (pre-NPZ) or a flat FEN CSV.

Usage
─────
  python maia_vs_krut_eval.py \\
      --krut-checkpoint checkpoints/best_v4.pt \\
      --data-dir dataset/v1/test \\
      --maia-weights maia2-1500.pb.gz \\
      --lc0-path ./lc0 \\
      --parquet-dir dataset/parquet/test \\
      --output-dir comparison_results \\
      --batch-size 256 \\
      --max-positions 0
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from chess_mimo_model_v4 import ChessMIMOModelV4
from mimo_dataset_polars import MIMOCompactDataset


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def elo_bucket(elo: float) -> str:
    """Return the 100-point bucket label, e.g. '1500-1599'."""
    lo = int(elo // 100) * 100
    return f"{lo}-{lo + 99}"


def stm_elo_from_tabular(tabular: np.ndarray) -> float:
    """Extract side-to-move Elo from a tabular vector.

    tabular[1] = white_elo / 3000
    tabular[2] = black_elo / 3000
    tabular[5] = 1.0 if side-to-move is White, else 0.0
    """
    w_elo = tabular[1] * 3000.0
    b_elo = tabular[2] * 3000.0
    return w_elo if tabular[5] > 0.5 else b_elo


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — KRUT inference (batch, via MIMOCompactDataset)
# ═══════════════════════════════════════════════════════════════════════

def load_krut_model(checkpoint_path: str, device: torch.device) -> ChessMIMOModelV4:
    """Load MIMO V4 checkpoint with registry-aware component versions."""
    print(f"[KRUT] Loading {checkpoint_path} …")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', ckpt.get('model_config', {}))
    comp_ver = cfg.get('component_versions', {})

    model = ChessMIMOModelV4(
        cnn_channels=cfg.get('cnn_channels', 128),
        num_res_blocks=cfg.get('res_blocks', 6),
        tabular_dim=18,
        max_possible=cfg.get('max_possible', 220),
        hidden_dim=cfg.get('hidden_dim', 256),
        mistake_expert_ver=comp_ver.get('mistake_expert', 'default'),
        time_expert_ver=comp_ver.get('time_expert', 'default'),
        wdl_expert_ver=comp_ver.get('wdl_expert', 'default'),
        move_head_ver=comp_ver.get('move_head', 'default'),
    ).to(device)

    state = ckpt.get('model_state_dict', ckpt)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    epoch = ckpt.get('epoch', '?')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[KRUT] Loaded epoch {epoch} — {n_params:,} params")
    print(f"[KRUT] Versions: {model.component_versions}")
    return model


@torch.no_grad()
def run_krut_inference(
    model: ChessMIMOModelV4,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run KRUT on the full test set.

    Returns:
        move_probs:  (N, M) — softmax move probabilities per position
        actual_idx:  (N,)   — index of the human's actual move
        stm_elos:    (N,)   — Elo of the side to move
    """
    model.eval()
    all_probs = []
    all_actual = []
    all_stm_elo = []

    t0 = time.time()
    for i, batch in enumerate(loader):
        cp = batch['current_planes'].to(device, non_blocking=True)
        pf = batch['possible_from_sq'].to(device, non_blocking=True)
        pt = batch['possible_to_sq'].to(device, non_blocking=True)
        pp = batch['possible_promo'].to(device, non_blocking=True)
        ps = batch['possible_scalars'].to(device, non_blocking=True)
        pm = batch['possible_mask'].to(device, non_blocking=True)
        tab = batch['tabular'].to(device, non_blocking=True)
        aidx = batch['actual_idx'].to(device, non_blocking=True)

        with autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(cp, pf, pt, pp, ps, pm, tab, actual_idx=aidx)

        probs = torch.softmax(outputs['move_logits'], dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_actual.append(batch['actual_idx'].numpy())

        # Extract STM Elo from tabular
        tab_np = batch['tabular'].numpy()
        stm_elo = np.where(tab_np[:, 5] > 0.5,
                           tab_np[:, 1] * 3000.0,
                           tab_np[:, 2] * 3000.0)
        all_stm_elo.append(stm_elo)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [KRUT] {i+1}/{len(loader)} batches  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    move_probs = np.concatenate(all_probs, axis=0)
    actual_idx = np.concatenate(all_actual, axis=0)
    stm_elos = np.concatenate(all_stm_elo, axis=0)
    print(f"[KRUT] Inference done: {len(actual_idx):,} positions in {elapsed:.1f}s")
    return move_probs, actual_idx, stm_elos


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — Maia-2 inference (via LC0 subprocess, UCI)
# ═══════════════════════════════════════════════════════════════════════

class LC0MaiaRunner:
    """Drives LC0 with Maia-2 weights via UCI to get policy probabilities.

    Uses VerboseMoveStats to extract the raw policy probability for every
    legal move at nodes=1.
    """

    def __init__(self, lc0_path: str, weights_path: str,
                 backend: str = 'cuda-fp16', threads: int = 1):
        self.proc = subprocess.Popen(
            [lc0_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send('uci')
        self._wait_for('uciok')

        # Configure for pure policy output at nodes=1
        opts = {
            'WeightsFile': weights_path,
            'Backend': backend,
            'Threads': str(threads),
            'VerboseMoveStats': 'true',
            'SmartPruningFactor': '0',
        }
        for k, v in opts.items():
            self._send(f'setoption name {k} value {v}')

        self._send('isready')
        self._wait_for('readyok')
        print(f"[MAIA] LC0 ready — weights: {Path(weights_path).name}, "
              f"backend: {backend}")

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + '\n')
        self.proc.stdin.flush()

    def _wait_for(self, token: str) -> List[str]:
        """Read lines until one starts with `token`.  Return all lines."""
        lines = []
        while True:
            line = self.proc.stdout.readline().strip()
            lines.append(line)
            if line.startswith(token):
                return lines

    def get_policy(self, fen: str) -> Dict[str, float]:
        """Get raw policy probabilities for all legal moves.

        Returns {uci_move: probability} (probabilities sum to ~1).
        """
        self._send(f'position fen {fen}')
        self._send('go nodes 1')

        move_probs: Dict[str, float] = {}
        lines = []
        while True:
            line = self.proc.stdout.readline().strip()
            lines.append(line)

            # VerboseMoveStats lines look like:
            #   info string e2e4  (N:  0) (+ 0) (P: 12.34%) (WL: ...) ...
            if line.startswith('info string') and '(P:' in line:
                m = re.match(
                    r'info string\s+(\S+)\s+.*\(P:\s*([\d.]+)%\)',
                    line,
                )
                if m:
                    uci_move = m.group(1)
                    pct = float(m.group(2))
                    move_probs[uci_move] = pct / 100.0

            if line.startswith('bestmove'):
                break

        # Normalise in case percentages don't sum to exactly 100
        total = sum(move_probs.values())
        if total > 0:
            move_probs = {m: p / total for m, p in move_probs.items()}

        return move_probs

    def close(self):
        try:
            self._send('quit')
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def load_parquet_positions(parquet_dir: str,
                           max_positions: int = 0) -> List[Dict]:
    """Load positions from parquet files for Maia-2 evaluation.

    Each parquet row must have at minimum:
      fen_before, move_uci (or actual_move), white_elo, black_elo, color

    Returns list of dicts with keys: fen, actual_move, stm_elo, color
    """
    try:
        import polars as pl
    except ImportError:
        import pandas as pd
        return _load_parquet_pandas(parquet_dir, max_positions)

    parquet_path = Path(parquet_dir)
    files = sorted(parquet_path.glob('**/*.parquet'))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {parquet_dir}")

    print(f"[MAIA] Loading positions from {len(files)} parquet files …")
    positions = []
    for pf in files:
        df = pl.read_parquet(pf)

        # Find the FEN column
        fen_col = 'fen_before' if 'fen_before' in df.columns else 'fen'
        if fen_col not in df.columns:
            print(f"  [WARN] Skipping {pf.name}: no fen column found")
            continue

        # Find the move column
        move_col = None
        for c in ['move_uci', 'actual_move', 'move']:
            if c in df.columns:
                move_col = c
                break
        if move_col is None:
            print(f"  [WARN] Skipping {pf.name}: no move column found")
            continue

        for row in df.iter_rows(named=True):
            fen = row[fen_col]
            move = row[move_col]
            w_elo = float(row.get('white_elo', 1500))
            b_elo = float(row.get('black_elo', 1500))

            # Determine side to move from FEN or color column
            if 'color' in row:
                color = str(row['color']).lower()
                is_white = color in ('white', 'w')
            else:
                # Parse from FEN
                is_white = ' w ' in fen

            stm_elo = w_elo if is_white else b_elo

            positions.append({
                'fen': fen,
                'actual_move': move,
                'stm_elo': stm_elo,
            })

            if 0 < max_positions <= len(positions):
                break
        if 0 < max_positions <= len(positions):
            break

    print(f"[MAIA] Loaded {len(positions):,} positions from parquet")
    return positions


def _load_parquet_pandas(parquet_dir: str,
                         max_positions: int = 0) -> List[Dict]:
    """Fallback parquet loader using pandas."""
    import pandas as pd

    parquet_path = Path(parquet_dir)
    files = sorted(parquet_path.glob('**/*.parquet'))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {parquet_dir}")

    print(f"[MAIA] Loading positions from {len(files)} parquet files (pandas) …")
    positions = []
    for pf in files:
        df = pd.read_parquet(pf)
        fen_col = 'fen_before' if 'fen_before' in df.columns else 'fen'
        move_col = next((c for c in ['move_uci', 'actual_move', 'move']
                         if c in df.columns), None)
        if fen_col not in df.columns or move_col is None:
            continue

        for _, row in df.iterrows():
            fen = row[fen_col]
            move = row[move_col]
            w_elo = float(row.get('white_elo', 1500))
            b_elo = float(row.get('black_elo', 1500))
            if 'color' in row:
                is_white = str(row['color']).lower() in ('white', 'w')
            else:
                is_white = ' w ' in fen
            stm_elo = w_elo if is_white else b_elo
            positions.append({
                'fen': fen,
                'actual_move': move,
                'stm_elo': stm_elo,
            })
            if 0 < max_positions <= len(positions):
                break
        if 0 < max_positions <= len(positions):
            break

    print(f"[MAIA] Loaded {len(positions):,} positions from parquet")
    return positions


def load_fen_csv(csv_path: str, max_positions: int = 0) -> List[Dict]:
    """Load positions from a flat CSV/TSV.

    Expected columns: fen, actual_move, stm_elo
    (or: fen, actual_move, white_elo, black_elo, color)
    """
    import csv
    positions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fen = row['fen']
            move = row.get('actual_move', row.get('move', ''))
            if 'stm_elo' in row:
                stm_elo = float(row['stm_elo'])
            else:
                w_elo = float(row.get('white_elo', 1500))
                b_elo = float(row.get('black_elo', 1500))
                color = row.get('color', '')
                is_white = color.lower() in ('white', 'w', '') and ' w ' in fen
                stm_elo = w_elo if is_white else b_elo
            positions.append({
                'fen': fen,
                'actual_move': move,
                'stm_elo': stm_elo,
            })
            if 0 < max_positions <= len(positions):
                break
    print(f"[MAIA] Loaded {len(positions):,} positions from CSV")
    return positions


def run_maia_inference(
    runner: LC0MaiaRunner,
    positions: List[Dict],
) -> Tuple[List[Dict[str, float]], List[str], np.ndarray]:
    """Run Maia-2 on all positions.

    Returns:
        policy_dists:  list of {uci_move: prob} dicts
        actual_moves:  list of actual move UCI strings
        stm_elos:      (N,) array of side-to-move Elo
    """
    N = len(positions)
    policy_dists = []
    actual_moves = []
    stm_elos = np.zeros(N, dtype=np.float32)

    t0 = time.time()
    for i, pos in enumerate(positions):
        policy = runner.get_policy(pos['fen'])
        policy_dists.append(policy)
        actual_moves.append(pos['actual_move'])
        stm_elos[i] = pos['stm_elo']

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate if rate > 0 else 0
            print(f"  [MAIA] {i+1:,}/{N:,}  "
                  f"({rate:.0f} pos/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"[MAIA] Inference done: {N:,} positions in {elapsed:.1f}s "
          f"({N / max(elapsed, 0.1):.0f} pos/s)")
    return policy_dists, actual_moves, stm_elos


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — Metrics
# ═══════════════════════════════════════════════════════════════════════

EPS = 1e-10  # floor for log to avoid -inf


def compute_krut_metrics(move_probs: np.ndarray,
                         actual_idx: np.ndarray,
                         stm_elos: np.ndarray) -> Dict:
    """Compute all KRUT metrics: overall + per-Elo-bucket.

    move_probs: (N, M) softmax probabilities
    actual_idx: (N,) index of the actual move
    stm_elos:   (N,) Elo of the side to move
    """
    results = {}
    N = len(actual_idx)

    # Mask out invalid positions (actual_idx == -1)
    valid = actual_idx >= 0
    mp = move_probs[valid]
    ai = actual_idx[valid]
    elos = stm_elos[valid]
    N_valid = int(valid.sum())

    # ── Overall ────────────────────────────────────────────────────
    overall = _bucket_metrics_krut(mp, ai, 'krut_overall')
    overall['krut_overall_n'] = N_valid
    results.update(overall)

    # ── Per 100-Elo bucket ─────────────────────────────────────────
    buckets = defaultdict(list)
    for i in range(N_valid):
        bk = elo_bucket(elos[i])
        buckets[bk].append(i)

    for bk in sorted(buckets.keys()):
        idxs = np.array(buckets[bk])
        prefix = f"krut_{bk}"
        bk_metrics = _bucket_metrics_krut(mp[idxs], ai[idxs], prefix)
        bk_metrics[f"{prefix}_n"] = len(idxs)
        results.update(bk_metrics)

    return results


def _bucket_metrics_krut(move_probs: np.ndarray,
                         actual_idx: np.ndarray,
                         prefix: str) -> Dict:
    """Compute top-k accuracy and log-loss for KRUT predictions."""
    N = len(actual_idx)
    if N == 0:
        return {}

    # Vectorised top-k: argsort descending along axis=1
    ranked = np.argsort(move_probs, axis=1)[:, ::-1]

    top1 = (ranked[:, 0] == actual_idx).sum()
    top2 = sum(actual_idx[i] in ranked[i, :2] for i in range(N))
    top3 = sum(actual_idx[i] in ranked[i, :3] for i in range(N))
    top5 = sum(actual_idx[i] in ranked[i, :5] for i in range(N))

    # Log-loss: -log(P(actual))
    actual_probs = move_probs[np.arange(N), actual_idx]
    log_loss = -np.log(np.clip(actual_probs, EPS, None)).mean()

    return {
        f'{prefix}_top1': top1 / N,
        f'{prefix}_top2': top2 / N,
        f'{prefix}_top3': top3 / N,
        f'{prefix}_top5': top5 / N,
        f'{prefix}_logloss': float(log_loss),
    }


def compute_maia_metrics(policy_dists: List[Dict[str, float]],
                         actual_moves: List[str],
                         stm_elos: np.ndarray) -> Dict:
    """Compute all Maia-2 metrics: overall + per-Elo-bucket."""
    results = {}
    N = len(actual_moves)

    # ── Overall ────────────────────────────────────────────────────
    overall = _bucket_metrics_maia(policy_dists, actual_moves, 'maia_overall')
    overall['maia_overall_n'] = N
    results.update(overall)

    # ── Per 100-Elo bucket ─────────────────────────────────────────
    buckets = defaultdict(list)
    for i in range(N):
        bk = elo_bucket(stm_elos[i])
        buckets[bk].append(i)

    for bk in sorted(buckets.keys()):
        idxs = buckets[bk]
        prefix = f"maia_{bk}"
        pd_sub = [policy_dists[i] for i in idxs]
        am_sub = [actual_moves[i] for i in idxs]
        bk_metrics = _bucket_metrics_maia(pd_sub, am_sub, prefix)
        bk_metrics[f"{prefix}_n"] = len(idxs)
        results.update(bk_metrics)

    return results


def _bucket_metrics_maia(policy_dists: List[Dict[str, float]],
                         actual_moves: List[str],
                         prefix: str) -> Dict:
    """Compute top-k accuracy and log-loss for Maia-2 predictions."""
    N = len(actual_moves)
    if N == 0:
        return {}

    top1 = top2 = top3 = top5 = 0
    log_losses = []

    for i in range(N):
        probs = policy_dists[i]
        actual = actual_moves[i]

        # Rank moves by probability descending
        ranked = sorted(probs.keys(), key=lambda m: probs[m], reverse=True)

        if len(ranked) >= 1 and ranked[0] == actual:
            top1 += 1
        if actual in ranked[:2]:
            top2 += 1
        if actual in ranked[:3]:
            top3 += 1
        if actual in ranked[:5]:
            top5 += 1

        p = probs.get(actual, EPS)
        log_losses.append(-math.log(max(p, EPS)))

    return {
        f'{prefix}_top1': top1 / N,
        f'{prefix}_top2': top2 / N,
        f'{prefix}_top3': top3 / N,
        f'{prefix}_top5': top5 / N,
        f'{prefix}_logloss': float(np.mean(log_losses)),
    }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — Report
# ═══════════════════════════════════════════════════════════════════════

def generate_report(all_metrics: Dict, output_dir: str, has_maia: bool):
    """Generate human-readable comparison report."""
    out = Path(output_dir)
    lines = []
    lines.append("=" * 80)
    lines.append("  Maia-2 vs KRUT — Move Prediction Accuracy Comparison")
    lines.append("=" * 80)
    lines.append("")

    # ── Overall ───────────────────────────────────────────────────
    lines.append("OVERALL")
    lines.append("-" * 80)

    header = f"  {'Metric':<20s}"
    header += f"  {'KRUT':>12s}"
    if has_maia:
        header += f"  {'Maia-2':>12s}"
    lines.append(header)
    lines.append(f"  {'─' * 20}  {'─' * 12}" +
                 (f"  {'─' * 12}" if has_maia else ""))

    krut_n = all_metrics.get('krut_overall_n', 0)
    maia_n = all_metrics.get('maia_overall_n', 0)
    lines.append(f"  {'positions':<20s}  {krut_n:>12,d}" +
                 (f"  {maia_n:>12,d}" if has_maia else ""))

    for metric in ['top1', 'top2', 'top3', 'top5', 'logloss']:
        kv = all_metrics.get(f'krut_overall_{metric}')
        k_str = f"{kv:.4f}" if kv is not None else "N/A"
        row = f"  {metric:<20s}  {k_str:>12s}"
        if has_maia:
            mv = all_metrics.get(f'maia_overall_{metric}')
            m_str = f"{mv:.4f}" if mv is not None else "N/A"
            row += f"  {m_str:>12s}"
        lines.append(row)
    lines.append("")

    # ── Per-Elo bucket ────────────────────────────────────────────
    lines.append("PER-ELO BUCKET (side-to-move rating, 100-point ranges)")
    lines.append("-" * 80)

    # Collect all bucket labels
    bucket_labels = set()
    for k in all_metrics:
        for prefix in ('krut_', 'maia_'):
            if k.startswith(prefix) and k.endswith('_n'):
                label = k[len(prefix):-2]
                if label != 'overall' and '-' in label:
                    bucket_labels.add(label)

    if bucket_labels:
        header = f"  {'Elo bucket':<14s}  {'n_KRUT':>8s}  {'top1':>7s}  {'top2':>7s}  {'top3':>7s}  {'top5':>7s}  {'logloss':>8s}"
        if has_maia:
            header += f"  │  {'n_Maia':>8s}  {'top1':>7s}  {'top2':>7s}  {'top3':>7s}  {'top5':>7s}  {'logloss':>8s}"
        lines.append(header)
        sep = f"  {'─' * 14}  {'─' * 8}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 8}"
        if has_maia:
            sep += f"  │  {'─' * 8}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 8}"
        lines.append(sep)

        for bk in sorted(bucket_labels, key=lambda x: int(x.split('-')[0])):
            kn = all_metrics.get(f'krut_{bk}_n', 0)
            row = f"  {bk:<14s}  {kn:>8,d}"
            for m in ['top1', 'top2', 'top3', 'top5']:
                v = all_metrics.get(f'krut_{bk}_{m}')
                row += f"  {v:>7.4f}" if v is not None else f"  {'N/A':>7s}"
            v = all_metrics.get(f'krut_{bk}_logloss')
            row += f"  {v:>8.4f}" if v is not None else f"  {'N/A':>8s}"

            if has_maia:
                mn = all_metrics.get(f'maia_{bk}_n', 0)
                row += f"  │  {mn:>8,d}"
                for m in ['top1', 'top2', 'top3', 'top5']:
                    v = all_metrics.get(f'maia_{bk}_{m}')
                    row += f"  {v:>7.4f}" if v is not None else f"  {'N/A':>7s}"
                v = all_metrics.get(f'maia_{bk}_logloss')
                row += f"  {v:>8.4f}" if v is not None else f"  {'N/A':>8s}"

            lines.append(row)
    lines.append("")
    lines.append("=" * 80)

    report = '\n'.join(lines)
    report_path = out / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    print(report)
    print(f"\n[REPORT] Saved → {report_path}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Maia-2 vs KRUT move-prediction accuracy comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # KRUT
    parser.add_argument('--krut-checkpoint', required=True,
                        help='Path to KRUT (MIMO V4) checkpoint .pt file')
    parser.add_argument('--data-dir', required=True,
                        help='NPZ shard directory for KRUT (test split)')

    # Maia-2 (optional — can run KRUT-only with --skip-maia)
    parser.add_argument('--maia-weights', type=str, default=None,
                        help='Path to Maia-2 LC0 weights (.pb.gz or .onnx)')
    parser.add_argument('--lc0-path', type=str, default='lc0',
                        help='Path to LC0 binary')
    parser.add_argument('--lc0-backend', type=str, default='cuda-fp16',
                        help='LC0 backend (default: cuda-fp16)')

    # Maia-2 position source
    parser.add_argument('--parquet-dir', type=str, default=None,
                        help='Parquet directory with FENs + moves for Maia '
                             '(pre-NPZ test data)')
    parser.add_argument('--fen-csv', type=str, default=None,
                        help='CSV file with columns: fen, actual_move, stm_elo '
                             '(alternative to --parquet-dir)')

    # General
    parser.add_argument('--skip-maia', action='store_true',
                        help='Skip Maia-2 inference (KRUT-only evaluation)')
    parser.add_argument('--output-dir', default='comparison_results')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-positions', type=int, default=0,
                        help='Cap on number of positions (0 = all)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--cache-shards', type=int, default=2)

    args = parser.parse_args()
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict = {}
    has_maia = False

    # ── 1. KRUT ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KRUT INFERENCE")
    print("=" * 60)

    model = load_krut_model(args.krut_checkpoint, device)

    ds = MIMOCompactDataset(args.data_dir, cache_shards=args.cache_shards)
    if args.max_positions > 0:
        from torch.utils.data import Subset
        n = min(args.max_positions, len(ds))
        ds = Subset(ds, list(range(n)))

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    move_probs, actual_idx, stm_elos = run_krut_inference(model, loader, device)
    krut_metrics = compute_krut_metrics(move_probs, actual_idx, stm_elos)
    all_metrics.update(krut_metrics)

    # Free GPU memory before Maia
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── 2. Maia-2 ─────────────────────────────────────────────────
    if not args.skip_maia:
        if args.maia_weights is None:
            parser.error("--maia-weights is required unless --skip-maia")
        if args.parquet_dir is None and args.fen_csv is None:
            parser.error("Provide --parquet-dir or --fen-csv for Maia positions "
                         "(FENs are not stored in NPZ shards)")

        print("\n" + "=" * 60)
        print("MAIA-2 INFERENCE")
        print("=" * 60)

        # Load positions
        if args.parquet_dir:
            positions = load_parquet_positions(
                args.parquet_dir, args.max_positions)
        else:
            positions = load_fen_csv(args.fen_csv, args.max_positions)

        if not positions:
            print("[MAIA] No positions loaded — skipping Maia evaluation")
        else:
            runner = LC0MaiaRunner(
                args.lc0_path, args.maia_weights,
                backend=args.lc0_backend,
            )
            try:
                policy_dists, actual_moves, maia_elos = run_maia_inference(
                    runner, positions)
                maia_metrics = compute_maia_metrics(
                    policy_dists, actual_moves, maia_elos)
                all_metrics.update(maia_metrics)
                has_maia = True
            finally:
                runner.close()
    else:
        print("\n[MAIA] Skipped (--skip-maia)")

    # ── 3. Save & report ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Save raw metrics JSON
    metrics_path = out / 'comparison_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(
            {k: (float(v) if isinstance(v, (np.floating, np.integer, float))
                 else int(v) if isinstance(v, (np.signedinteger, int))
                 else v)
             for k, v in sorted(all_metrics.items())},
            f, indent=2,
        )
    print(f"[SAVE] Metrics → {metrics_path}")

    # Generate text report
    generate_report(all_metrics, str(out), has_maia)

    print(f"\n✓ Evaluation complete → {out}")


if __name__ == '__main__':
    main()
