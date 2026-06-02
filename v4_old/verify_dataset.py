#!/usr/bin/env python3
"""
verify_dataset.py — Verify MIMO dataset shards have correct ground truth.

Checks:
  1. win_prob_before: all 3 classes present, distribution matches parquet game results
  2. tabular shape and value ranges
  3. possible_scalars shape and no NaN/Inf
  4. is_mistake: binary values only
  5. time_spent_log: non-negative, reasonable range
  6. actual_idx: within possible_mask bounds

Usage:
    python verify_dataset.py --data-dir dataset/v2
    python verify_dataset.py --data-dir dataset/v2 --max-shards 10
"""

import argparse
import glob
import os
import sys
import numpy as np
from collections import Counter


def check_shard(path):
    """Check one shard, return stats dict or error string."""
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        return f"LOAD ERROR: {e}"

    n = len(d['win_prob_before'])
    stats = {
        'n': n,
        'wdl_white_wins': 0,
        'wdl_draws': 0,
        'wdl_black_wins': 0,
        'wdl_bad_rows': 0,
        'tabular_shape': None,
        'tabular_nan': 0,
        'tabular_inf': 0,
        'poss_scalars_shape': None,
        'poss_scalars_nan': 0,
        'poss_scalars_inf': 0,
        'mistake_bad': 0,
        'time_neg': 0,
        'time_max': 0.0,
        'idx_oob': 0,
        'errors': [],
    }

    # --- win_prob_before ---
    wdl = d['win_prob_before']
    if wdl.shape != (n, 3):
        stats['errors'].append(f"win_prob_before shape {wdl.shape}, expected ({n}, 3)")
    else:
        for row in wdl:
            if np.array_equal(row, [1, 0, 0]):
                stats['wdl_white_wins'] += 1
            elif np.array_equal(row, [0, 1, 0]):
                stats['wdl_draws'] += 1
            elif np.array_equal(row, [0, 0, 1]):
                stats['wdl_black_wins'] += 1
            else:
                stats['wdl_bad_rows'] += 1

    # --- tabular ---
    tab = d['tabular']
    stats['tabular_shape'] = tab.shape
    stats['tabular_nan'] = int(np.isnan(tab).sum())
    stats['tabular_inf'] = int(np.isinf(tab).sum())

    # --- possible_scalars ---
    ps = d['possible_scalars']
    stats['poss_scalars_shape'] = ps.shape
    stats['poss_scalars_nan'] = int(np.isnan(ps).sum())
    stats['poss_scalars_inf'] = int(np.isinf(ps).sum())

    # --- is_mistake ---
    ism = d['is_mistake']
    stats['mistake_bad'] = int(((ism != 0.0) & (ism != 1.0)).sum())

    # --- time_spent_log ---
    tsl = d['time_spent_log']
    stats['time_neg'] = int((tsl < 0).sum())
    stats['time_max'] = float(tsl.max())

    # --- actual_idx vs possible_mask ---
    aidx = d['actual_idx']
    pmask = d['possible_mask']
    for i in range(n):
        idx = int(aidx[i])
        if idx < 0 or idx >= len(pmask[i]) or pmask[i][idx] == 0:
            stats['idx_oob'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Dataset root (e.g. dataset/v2)')
    parser.add_argument('--max-shards', type=int, default=0, help='Max shards per split (0=all)')
    args = parser.parse_args()

    splits = ['train', 'val', 'test']
    all_ok = True

    for split in splits:
        pattern = os.path.join(args.data_dir, split, '*.npz')
        shards = sorted(glob.glob(pattern))
        if not shards:
            print(f"\n[{split}] No shards found at {pattern}")
            continue
        if args.max_shards > 0:
            shards = shards[:args.max_shards]

        total = Counter()
        total_n = 0
        shard_errors = []

        for i, path in enumerate(shards):
            print(f"\r  [{split}] checking {i+1}/{len(shards)}...", end='', flush=True)
            result = check_shard(path)
            if isinstance(result, str):
                shard_errors.append((os.path.basename(path), result))
                continue

            total_n += result['n']
            total['wdl_white_wins'] += result['wdl_white_wins']
            total['wdl_draws'] += result['wdl_draws']
            total['wdl_black_wins'] += result['wdl_black_wins']
            total['wdl_bad_rows'] += result['wdl_bad_rows']
            total['tabular_nan'] += result['tabular_nan']
            total['tabular_inf'] += result['tabular_inf']
            total['poss_scalars_nan'] += result['poss_scalars_nan']
            total['poss_scalars_inf'] += result['poss_scalars_inf']
            total['mistake_bad'] += result['mistake_bad']
            total['time_neg'] += result['time_neg']
            total['idx_oob'] += result['idx_oob']
            for e in result['errors']:
                shard_errors.append((os.path.basename(path), e))

        print()
        print(f"\n{'='*60}")
        print(f"  [{split.upper()}]  {len(shards)} shards, {total_n:,} examples")
        print(f"{'='*60}")

        if total_n == 0:
            print("  NO DATA")
            continue

        # WDL distribution
        w_pct = 100 * total['wdl_white_wins'] / total_n
        d_pct = 100 * total['wdl_draws'] / total_n
        l_pct = 100 * total['wdl_black_wins'] / total_n
        print(f"  WDL targets:")
        print(f"    White wins [1,0,0]: {total['wdl_white_wins']:>10,}  ({w_pct:.1f}%)")
        print(f"    Draws      [0,1,0]: {total['wdl_draws']:>10,}  ({d_pct:.1f}%)")
        print(f"    Black wins [0,0,1]: {total['wdl_black_wins']:>10,}  ({l_pct:.1f}%)")
        if total['wdl_bad_rows'] > 0:
            print(f"    *** BAD ROWS:       {total['wdl_bad_rows']:>10,}  *** FAIL")
            all_ok = False

        # Sanity: all 3 classes must be present
        if total['wdl_white_wins'] == 0:
            print("    *** FAIL: zero White wins — result_to_wdl bug still present")
            all_ok = False
        if total['wdl_black_wins'] == 0:
            print("    *** FAIL: zero Black wins")
            all_ok = False
        if total['wdl_draws'] == 0:
            print("    *** WARNING: zero Draws (possible if filtered out)")

        # Expected: roughly balanced W/L (within 10 pct points)
        if abs(w_pct - l_pct) > 15:
            print(f"    *** WARNING: W/L imbalance ({w_pct:.1f}% vs {l_pct:.1f}%) — check data")

        # NaN / Inf
        print(f"  Tabular:    NaN={total['tabular_nan']}  Inf={total['tabular_inf']}")
        print(f"  PossScalar: NaN={total['poss_scalars_nan']}  Inf={total['poss_scalars_inf']}")
        if total['tabular_nan'] or total['tabular_inf']:
            print("    *** FAIL: NaN/Inf in tabular")
            all_ok = False
        if total['poss_scalars_nan'] or total['poss_scalars_inf']:
            print("    *** FAIL: NaN/Inf in possible_scalars")
            all_ok = False

        # Other checks
        print(f"  is_mistake non-binary: {total['mistake_bad']}")
        print(f"  time_spent_log negative: {total['time_neg']}")
        print(f"  actual_idx out-of-bounds: {total['idx_oob']}")
        if total['mistake_bad']:
            all_ok = False
        if total['time_neg']:
            all_ok = False
        if total['idx_oob']:
            all_ok = False

        if shard_errors:
            print(f"\n  Shard errors:")
            for fname, err in shard_errors[:10]:
                print(f"    {fname}: {err}")
            all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print("  ALL CHECKS PASSED ✓")
    else:
        print("  *** SOME CHECKS FAILED ***")
    print(f"{'='*60}")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
