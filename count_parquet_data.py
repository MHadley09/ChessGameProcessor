"""
count_parquet_data.py — Diagnostic script for verifying parquet output.

Reads parquet files from output/parquet/ and prints:
- Row counts for games, moves, possible_moves
- Column names to verify schema correctness
- Sample values from first few rows
"""

import sys
import os
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    print("ERROR: pyarrow not installed. Run: pip install pyarrow")
    sys.exit(1)


def read_all_parquet(directory: Path) -> pa.Table:
    """Read all parquet files in a directory into one table."""
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return None
    tables = [pq.read_table(str(f)) for f in files]
    return pa.concat_tables(tables)


def print_table_info(name: str, table: pa.Table, sample_rows: int = 3):
    """Print summary info for a parquet table."""
    print(f"\n{'='*60}")
    print(f"  {name}: {table.num_rows:,} rows, {table.num_columns} columns")
    print(f"{'='*60}")
    print(f"  Columns: {table.column_names}")

    if table.num_rows > 0 and sample_rows > 0:
        n = min(sample_rows, table.num_rows)
        print(f"\n  First {n} rows (sample values):")
        for i in range(n):
            print(f"  --- Row {i} ---")
            for col_name in table.column_names:
                val = table.column(col_name)[i].as_py()
                # Truncate long strings
                if isinstance(val, str) and len(val) > 80:
                    val = val[:80] + "..."
                print(f"    {col_name}: {val}")


def check_schema_match(name: str, table: pa.Table, expected_schema: pa.Schema):
    """Compare actual columns to expected schema."""
    actual_cols = set(table.column_names)
    expected_cols = set(f.name for f in expected_schema)

    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols

    if missing:
        print(f"\n  ⚠️  {name} MISSING columns: {sorted(missing)}")
    if extra:
        print(f"\n  ⚠️  {name} EXTRA columns: {sorted(extra)}")
    if not missing and not extra:
        print(f"\n  ✅ {name} columns match schema perfectly")


def check_wdl_values(name: str, table: pa.Table):
    """Check if WDL columns have non-zero values."""
    wdl_cols = [c for c in table.column_names if 'win_perc' in c or 'draw_perc' in c]
    if not wdl_cols:
        return

    print(f"\n  WDL check for {name}:")
    for col in wdl_cols:
        arr = table.column(col)
        non_null = arr.drop_null()
        non_zero = sum(1 for v in non_null if v.as_py() != 0.0)
        print(f"    {col}: {non_zero}/{len(non_null)} non-zero values "
              f"({'✅' if non_zero > 0 else '⚠️  ALL ZERO'})")


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "output/parquet"
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"ERROR: Directory not found: {base_path}")
        sys.exit(1)

    print(f"Scanning: {base_path.resolve()}")

    # Import canonical schemas for comparison
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from parquet_schema import GAMES_SCHEMA, MOVES_SCHEMA, POSSIBLE_MOVES_SCHEMA
        has_schemas = True
    except ImportError:
        has_schemas = False
        print("(parquet_schema.py not found — skipping schema comparison)")

    # Walk all subdirectories looking for games/moves/possible_moves
    found_any = False
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)

        for table_name, schema in [
            ("games", GAMES_SCHEMA if has_schemas else None),
            ("moves", MOVES_SCHEMA if has_schemas else None),
            ("possible_moves", POSSIBLE_MOVES_SCHEMA if has_schemas else None),
        ]:
            table_dir = root_path / table_name
            if table_dir.is_dir():
                table = read_all_parquet(table_dir)
                if table is not None:
                    found_any = True
                    rel_path = table_dir.relative_to(base_path)
                    print_table_info(f"{rel_path}", table)
                    if schema:
                        check_schema_match(str(rel_path), table, schema)
                    check_wdl_values(str(rel_path), table)

    if not found_any:
        print("\nNo parquet data found. Expected structure:")
        print("  output/parquet/<engine>/<version>/games/*.parquet")
        print("  output/parquet/<engine>/<version>/moves/*.parquet")
        print("  output/parquet/<engine>/<version>/possible_moves/*.parquet")


if __name__ == "__main__":
    main()
