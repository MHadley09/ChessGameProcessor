#!/usr/bin/env python3
"""
fix_wdl_perspective.py — Fix the WDL perspective bug in existing parquet data.
================================================================================

THE BUG
-------
batch_evaluator._extract_wdl() reads wdl.wins/draws/losses from python-chess's
PovWdl, which gives SIDE-TO-MOVE (STM) perspective.  But parallel_processor's
_wdl_percentages() labels the output as White's perspective (white_win_perc,
black_win_perc).  This is correct when White is STM, but SWAPPED when Black is
STM.

WHAT THIS SCRIPT FIXES (deterministically correct)
---------------------------------------------------
1. POSSIBLE_MOVES parquet: white_win_perc / black_win_perc
   - These ALWAYS come from multipv at the current position → STM = color field.
   - For color='Black': swap white_win_perc ↔ black_win_perc.
   - For color='White': already correct.

2. MOVES parquet: white_win_perc_before / black_win_perc_before (PARTIAL FIX)
   - Move 1 of each game (always White, fresh eval, STM=White): correct, no swap.
   - Black-to-move eval_before (cached from previous White's eval_after, which
     came from multipv STM=White): correct, no swap needed.
   - White-to-move eval_before for move_no > 1 (cached from previous Black's
     eval_after, which came from multipv STM=Black): SWAPPED, needs fix.
   - HOWEVER: we can't distinguish cached vs fresh eval_before from parquet
     alone.  For safety, this script fixes possible_moves (deterministic) and
     offers an OPTIONAL heuristic for moves eval_before.

3. MOVES parquet: white_win_perc_after / black_win_perc_after
   - These are READ by the dataset for the WDL "after" target?  Actually NO —
     the wdl_after target comes from result_to_wdl(game_result), not the
     parquet eval_after columns.  So eval_after errors don't affect training.
   - The eval_after IS used for caching though, propagating to the next move's
     eval_before.  Fixing eval_after in parquet doesn't help because the cache
     chain already baked the error into eval_before values at write time.

RECOMMENDATION FOR V4
---------------------
- Fix batch_evaluator._extract_wdl() to use .white() explicitly
- Re-run the pipeline for fully clean data
- Then the dataset loader doesn't need any workarounds

Usage:
    python fix_wdl_perspective.py <parquet_dir> [--output <output_dir>] [--fix-moves-before] [--dry-run]

    --fix-moves-before : Also apply the heuristic fix for moves eval_before
                         (swap White's move_no > 1 rows).  This is ~95% correct
                         for the majority-cache path but not guaranteed.
    --dry-run          : Print stats without writing any files.
"""

import argparse
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    import numpy as np
except ImportError:
    print("Requires: pip install pyarrow numpy")
    sys.exit(1)


def fix_possible_moves_table(table: pa.Table) -> tuple[pa.Table, dict]:
    """Swap white_win_perc ↔ black_win_perc for Black-to-move rows."""
    stats = {"total": len(table), "black_rows": 0, "swapped": 0, "null_skipped": 0}

    if len(table) == 0:
        return table, stats

    color = table.column("color").to_pylist()
    w_win = table.column("white_win_perc").to_pylist()
    b_win = table.column("black_win_perc").to_pylist()

    new_w_win = list(w_win)
    new_b_win = list(b_win)

    for i in range(len(table)):
        if str(color[i]).strip().lower() == "black":
            stats["black_rows"] += 1
            if w_win[i] is not None and b_win[i] is not None:
                new_w_win[i] = b_win[i]
                new_b_win[i] = w_win[i]
                stats["swapped"] += 1
            else:
                stats["null_skipped"] += 1

    # Replace columns in the table
    col_idx_w = table.schema.get_field_index("white_win_perc")
    col_idx_b = table.schema.get_field_index("black_win_perc")
    table = table.set_column(col_idx_w, "white_win_perc", pa.array(new_w_win, type=pa.float64()))
    table = table.set_column(col_idx_b, "black_win_perc", pa.array(new_b_win, type=pa.float64()))
    return table, stats


def fix_moves_table(table: pa.Table, fix_before: bool = False) -> tuple[pa.Table, dict]:
    """
    Optionally swap white_win_perc_before ↔ black_win_perc_before for
    White-to-move rows with move_no > 1 (heuristic: these came from cache
    of previous Black's eval_after, which had STM=Black → swapped).
    """
    stats = {"total": len(table), "white_move_gt1": 0, "swapped_before": 0}

    if not fix_before or len(table) == 0:
        return table, stats

    color = table.column("color").to_pylist()
    move_no = table.column("move_no").to_pylist()
    w_before = table.column("white_win_perc_before").to_pylist()
    b_before = table.column("black_win_perc_before").to_pylist()

    new_w_before = list(w_before)
    new_b_before = list(b_before)

    for i in range(len(table)):
        c = str(color[i]).strip().lower()
        mn = int(move_no[i]) if move_no[i] is not None else 1

        # Heuristic: White's eval_before for move_no > 1 came from cache of
        # Black's eval_after (multipv STM=Black → swapped).
        # Move 1 (White) is fresh eval (STM=White → correct).
        if c == "white" and mn > 1:
            stats["white_move_gt1"] += 1
            if w_before[i] is not None and b_before[i] is not None:
                new_w_before[i] = b_before[i]
                new_b_before[i] = w_before[i]
                stats["swapped_before"] += 1

    col_w = table.schema.get_field_index("white_win_perc_before")
    col_b = table.schema.get_field_index("black_win_perc_before")
    table = table.set_column(col_w, "white_win_perc_before", pa.array(new_w_before, type=pa.float64()))
    table = table.set_column(col_b, "black_win_perc_before", pa.array(new_b_before, type=pa.float64()))
    return table, stats


def find_parquet_files(base_dir: Path) -> dict[str, list[Path]]:
    """Group parquet files by type (games, moves, possible_moves)."""
    found = {"moves": [], "possible_moves": [], "games": []}
    for f in sorted(base_dir.rglob("*.parquet")):
        name = f.stem.lower()
        if "possible" in name:
            found["possible_moves"].append(f)
        elif "move" in name:
            found["moves"].append(f)
        elif "game" in name:
            found["games"].append(f)
    return found


def invalidate_npy_cache(dataset_dir: Path):
    """Remove .npy_cache directories so training rebuilds from corrected parquet."""
    import shutil
    count = 0
    for cache_dir in dataset_dir.rglob(".npy_cache"):
        if cache_dir.is_dir():
            print(f"  Removing {cache_dir}")
            shutil.rmtree(cache_dir)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Fix WDL perspective bug in parquet data")
    parser.add_argument("parquet_dir", help="Directory containing parquet files (searched recursively)")
    parser.add_argument("--output", help="Output directory (default: overwrite in place)")
    parser.add_argument("--fix-moves-before", action="store_true",
                        help="Also fix moves eval_before (heuristic, ~95%% correct)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing")
    parser.add_argument("--invalidate-npy", default=None,
                        help="Path to dataset dir to invalidate .npy_cache (e.g. dataset/v1)")
    args = parser.parse_args()

    base = Path(args.parquet_dir)
    out_dir = Path(args.output) if args.output else base

    if not base.exists():
        print(f"Error: {base} does not exist")
        sys.exit(1)

    files = find_parquet_files(base)

    print(f"Found: {len(files['possible_moves'])} possible_moves, "
          f"{len(files['moves'])} moves, {len(files['games'])} games files\n")

    # ── Fix possible_moves ──────────────────────────────────────────────
    total_pm_stats = {"total": 0, "black_rows": 0, "swapped": 0, "null_skipped": 0}
    for f in files["possible_moves"]:
        print(f"Processing {f.name}...", end=" ", flush=True)
        table = pq.read_table(f)
        table, stats = fix_possible_moves_table(table)
        for k in total_pm_stats:
            total_pm_stats[k] += stats[k]
        print(f"{stats['swapped']:,} swapped / {stats['black_rows']:,} black rows")

        if not args.dry_run:
            out_path = out_dir / f.relative_to(base)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out_path)

    print(f"\n  Possible moves totals: {total_pm_stats['total']:,} rows, "
          f"{total_pm_stats['black_rows']:,} black, "
          f"{total_pm_stats['swapped']:,} swapped, "
          f"{total_pm_stats['null_skipped']:,} null-skipped\n")

    # ── Fix moves (optional) ────────────────────────────────────────────
    total_mv_stats = {"total": 0, "white_move_gt1": 0, "swapped_before": 0}
    for f in files["moves"]:
        print(f"Processing {f.name}...", end=" ", flush=True)
        table = pq.read_table(f)
        table, stats = fix_moves_table(table, fix_before=args.fix_moves_before)
        for k in total_mv_stats:
            total_mv_stats[k] += stats[k]
        if args.fix_moves_before:
            print(f"{stats['swapped_before']:,} swapped / {stats['white_move_gt1']:,} white move_no>1")
        else:
            print("(eval_before fix skipped — use --fix-moves-before)")

        if not args.dry_run:
            out_path = out_dir / f.relative_to(base)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out_path)

    if args.fix_moves_before:
        print(f"\n  Moves totals: {total_mv_stats['total']:,} rows, "
              f"{total_mv_stats['white_move_gt1']:,} white move_no>1, "
              f"{total_mv_stats['swapped_before']:,} swapped\n")

    # ── Invalidate .npy_cache ───────────────────────────────────────────
    if args.invalidate_npy and not args.dry_run:
        npy_dir = Path(args.invalidate_npy)
        if npy_dir.exists():
            n = invalidate_npy_cache(npy_dir)
            print(f"Invalidated {n} .npy_cache directories under {npy_dir}")

    if args.dry_run:
        print("\n*** DRY RUN — no files written ***")
    else:
        print(f"\nDone. Fixed parquet written to: {out_dir}")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("  1. Delete .npy_cache under your dataset dir so training rebuilds")
    print("  2. Fix batch_evaluator._extract_wdl() to use .white() for new runs")
    print("  3. Fix mimo_dataset_polars.py column reads (see TODOs in that file)")
    print("  4. For fully clean data: re-run pipeline with fixed batch_evaluator")
    print("=" * 70)


if __name__ == "__main__":
    main()
