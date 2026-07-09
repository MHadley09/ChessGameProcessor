#!/usr/bin/env python3
"""
repair_dataset.py — Repair broken dataset shards and dedup database.

Fixes two problems caused by the silent engine failure bug:
  1. Shard examples with no real evaluation data (engine was dead)
  2. Ghost entries in chessv5.db (games marked complete but produced
     zero shard examples because engine was dead)

Usage:
  # Dry run — report what would be fixed
  python repair_dataset.py <dataset_dir> --db <chessv5.db> --dry-run

  # Actually fix everything
  python repair_dataset.py <dataset_dir> --db <chessv5.db>

  # Just fix the DB (skip shard scan)
  python repair_dataset.py <dataset_dir> --db <chessv5.db> --db-only

  # Just scan shards (skip DB)
  python repair_dataset.py <dataset_dir> --db <chessv5.db> --shards-only
"""

import os
import sys
import argparse
import sqlite3
import time as time_module
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


# ── Shard scanning (vectorized) ─────────────────────────────────────────────

def scan_shard(shard_path):
    """Scan a single shard for broken examples using vectorized numpy ops.
    Returns (n_total, broken_mask) where broken_mask is a boolean array.
    """
    data = np.load(str(shard_path), allow_pickle=True)

    if 'possible_mask' not in data:
        return 0, np.array([], dtype=bool)

    possible_mask = data['possible_mask']     # (N, max_possible)
    actual_idx = data['actual_idx']           # (N,)
    poss_scalars = data['possible_scalars']   # (N, max_possible, D)
    tabular = data['tabular']                 # (N, T)

    n = len(possible_mask)

    # Legal move count per example
    n_legal = possible_mask.sum(axis=1)  # (N,)

    # Check 1: no legal moves at all
    no_legal = (n_legal == 0)

    # Check 2: actual_idx out of range
    bad_idx = (actual_idx < 0) | (actual_idx >= n_legal)

    # Check 3: all evals zero for non-padding rows
    # eval is column 0 of possible_scalars
    # For each example, check if ALL evals in the valid range are 0
    eval_col = poss_scalars[:, :, 0]  # (N, max_possible)
    # Mask out padding positions, then check if all valid evals are 0
    masked_evals = np.where(possible_mask.astype(bool), eval_col, np.nan)
    all_evals_zero = np.nansum(np.abs(masked_evals), axis=1) == 0
    # Only flag if n_legal > 1 (single legal move having 0 eval is fine)
    all_evals_zero = all_evals_zero & (n_legal > 1)

    # Check 4: all nodes zero (column 4 = log1p(nodes)/20)
    nodes_col = poss_scalars[:, :, 4]  # (N, max_possible)
    masked_nodes = np.where(possible_mask.astype(bool), nodes_col, np.nan)
    all_nodes_zero = np.nansum(np.abs(masked_nodes), axis=1) == 0
    all_nodes_zero = all_nodes_zero & (n_legal > 1)

    # Check 5: NaN/Inf in tabular
    bad_tabular = np.any(~np.isfinite(tabular), axis=1)

    # Check 6: NaN/Inf in active possible_scalars
    # Check all columns for valid (non-padding) rows
    finite_check = np.isfinite(poss_scalars)  # (N, max_possible, D)
    # Mask: only care about rows where possible_mask is 1
    pm_expanded = possible_mask.astype(bool)[:, :, np.newaxis]  # (N, max_possible, 1)
    # Where mask is True, check finite; where mask is False, treat as fine
    bad_scalars_per_pos = ~finite_check & pm_expanded  # (N, max_possible, D)
    bad_scalars = np.any(bad_scalars_per_pos.reshape(n, -1), axis=1)

    # Combine all checks
    broken = no_legal | bad_idx | all_evals_zero | all_nodes_zero | bad_tabular | bad_scalars

    return n, broken


def scan_and_fix_shards(dataset_dir, dry_run=False):
    """Scan all shards for broken examples. Remove them if not dry_run."""
    dataset_dir = Path(dataset_dir)
    total_scanned = 0
    total_broken = 0
    total_removed = 0

    for split in ('train', 'val', 'test'):
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        shards = sorted(split_dir.glob("*.npz"))
        if not shards:
            continue

        print(f"\n  {split}: {len(shards)} shards")
        split_broken = 0
        split_total = 0

        for si, shard_path in enumerate(shards):
            t0 = time_module.time()
            try:
                n, broken_mask = scan_shard(shard_path)
            except Exception as e:
                print(f"    [{si+1}/{len(shards)}] ERROR {shard_path.name}: {e}")
                continue

            n_broken = int(broken_mask.sum())
            elapsed = time_module.time() - t0
            split_total += n
            split_broken += n_broken

            status = f"{n_broken} broken" if n_broken > 0 else "clean"
            print(f"    [{si+1}/{len(shards)}] {shard_path.name}: {n:,} examples, {status} ({elapsed:.1f}s)")

            if n_broken > 0 and not dry_run:
                if n_broken == n:
                    os.remove(shard_path)
                    total_removed += n
                    print(f"             → deleted entire shard")
                else:
                    keep = ~broken_mask
                    data = np.load(str(shard_path), allow_pickle=True)
                    new_data = {key: data[key][keep] for key in data.files}
                    tmp = str(shard_path) + '.tmp'
                    np.savez_compressed(tmp, **new_data)
                    # np.savez_compressed may or may not append .npz
                    actual_tmp = tmp + '.npz' if os.path.exists(tmp + '.npz') else tmp
                    os.replace(actual_tmp, str(shard_path))
                    total_removed += n_broken
                    print(f"             → removed {n_broken}, kept {n - n_broken}")
            elif n_broken > 0:
                total_removed += n_broken

        total_scanned += split_total
        total_broken += split_broken
        print(f"    {split} total: {split_total:,} examples, {split_broken:,} broken")

    return total_scanned, total_broken, total_removed


# ── Database repair ──────────────────────────────────────────────────────────

def analyze_and_fix_db(db_path, dry_run=False, rate_threshold=10.0, window_secs=30):
    """Find and remove ghost entries by detecting processing rate spike.

    Normal: ~3.5-5 g/s (engine doing real work)
    Ghost: ~20-30 g/s (engine dead, just parsing PGN)
    """
    conn = sqlite3.connect(db_path)

    rows = conn.execute(
        "SELECT game_hash, processed_at FROM processed_games ORDER BY processed_at"
    ).fetchall()

    total = len(rows)
    if total == 0:
        print("  Database is empty")
        conn.close()
        return 0, 0, 0

    timestamps = [r[1] for r in rows]
    hashes = [r[0] for r in rows]

    first_ts, last_ts = timestamps[0], timestamps[-1]
    duration = last_ts - first_ts
    overall_rate = total / duration if duration > 0 else 0

    first_dt = datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    last_dt = datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    print(f"  Total entries: {total:,}")
    print(f"  Time range: {first_dt} → {last_dt}")
    print(f"  Duration: {duration/3600:.1f} hours")
    print(f"  Overall rate: {overall_rate:.1f} games/sec")

    # Find inflection: first window where rate > threshold
    print(f"\n  Scanning for rate spike (>{rate_threshold} g/s in {window_secs}s window)...")

    cutoff_idx = None
    window_start = 0
    for window_end in range(1, total):
        while timestamps[window_end] - timestamps[window_start] > window_secs:
            window_start += 1

        games_in_window = window_end - window_start + 1
        time_span = timestamps[window_end] - timestamps[window_start]

        if time_span > 0 and games_in_window >= 10:
            rate = games_in_window / time_span
            if rate > rate_threshold:
                cutoff_idx = window_start
                break

    if cutoff_idx is None:
        print(f"  ✅ No rate spike found — all entries look legitimate")
        conn.close()
        return total, 0, 0

    ghost_count = total - cutoff_idx
    good_count = cutoff_idx
    cutoff_ts = timestamps[cutoff_idx]
    cutoff_dt = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    if cutoff_idx > 1:
        pre_duration = timestamps[cutoff_idx - 1] - timestamps[0]
        pre_rate = cutoff_idx / pre_duration if pre_duration > 0 else 0
    else:
        pre_rate = 0

    post_duration = timestamps[-1] - cutoff_ts
    post_rate = ghost_count / post_duration if post_duration > 0 else 0

    print(f"\n  ⚠️  RATE SPIKE at {cutoff_dt}")
    print(f"     Good entries:  {good_count:,} at {pre_rate:.1f} g/s")
    print(f"     Ghost entries: {ghost_count:,} at {post_rate:.1f} g/s")

    if dry_run:
        print(f"\n  DRY RUN — would remove {ghost_count:,} ghost entries")
        conn.close()
        return total, ghost_count, 0

    # Delete ghosts
    print(f"\n  Removing {ghost_count:,} ghost entries...")
    ghost_hashes = hashes[cutoff_idx:]
    batch_size = 500
    removed = 0
    for i in range(0, len(ghost_hashes), batch_size):
        batch = ghost_hashes[i:i + batch_size]
        placeholders = ','.join('?' * len(batch))
        conn.execute(
            f"DELETE FROM processed_games WHERE game_hash IN ({placeholders})",
            batch,
        )
        removed += len(batch)
        if removed % 10000 == 0 or removed == len(ghost_hashes):
            print(f"    {removed:,}/{ghost_count:,}")

    conn.commit()
    remaining = conn.execute("SELECT COUNT(*) FROM processed_games").fetchone()[0]
    print(f"  ✅ Done. DB: {remaining:,} entries (was {total:,})")
    conn.close()
    return total, ghost_count, removed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Repair dataset after silent engine failure")
    parser.add_argument("dataset_dir", help="Root dataset directory (train/val/test)")
    parser.add_argument("--db", required=True, help="Path to chessv5.db")
    parser.add_argument("--dry-run", action="store_true", help="Report without modifying")
    parser.add_argument("--db-only", action="store_true", help="Only fix the DB")
    parser.add_argument("--shards-only", action="store_true", help="Only scan/fix shards")
    parser.add_argument("--rate-threshold", type=float, default=10.0,
                        help="g/s threshold for ghost detection (default: 10)")
    parser.add_argument("--window", type=int, default=30,
                        help="Sliding window in seconds (default: 30)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"ERROR: {dataset_dir} does not exist")
        sys.exit(1)

    mode = "DRY RUN" if args.dry_run else "REPAIR"
    print(f"{'='*60}")
    print(f"Dataset Repair — {mode}")
    print(f"{'='*60}")

    if not args.db_only:
        print(f"\nPHASE 1: Shard scan")
        t0 = time_module.time()
        total, broken, removed = scan_and_fix_shards(dataset_dir, dry_run=args.dry_run)
        elapsed = time_module.time() - t0
        print(f"\n  Scanned {total:,} examples in {elapsed:.1f}s")
        if total > 0:
            print(f"  Broken: {broken:,} ({broken/total*100:.2f}%)")
        if args.dry_run:
            print(f"  Would remove: {removed:,}")
        else:
            print(f"  Removed: {removed:,}")

    if not args.shards_only:
        if not os.path.exists(args.db):
            print(f"\nERROR: DB file {args.db} not found")
            sys.exit(1)
        print(f"\nPHASE 2: DB repair")
        t0 = time_module.time()
        analyze_and_fix_db(args.db, dry_run=args.dry_run,
                           rate_threshold=args.rate_threshold,
                           window_secs=args.window)
        print(f"  ({time_module.time() - t0:.1f}s)")

    print(f"\n{'='*60}")
    if args.dry_run:
        print("DRY RUN — no changes. Remove --dry-run to apply.")
    else:
        print("Done. Safe to restart pipeline.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
