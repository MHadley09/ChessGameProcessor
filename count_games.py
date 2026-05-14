"""Count unique game_ids across all parquet output folders."""
import os
import sys
import pyarrow.dataset as ds
import pyarrow as pa

base = sys.argv[1] if len(sys.argv) > 1 else "output"

all_ids = {}

for root, dirs, files in os.walk(base):
    parquets = [f for f in files if f.endswith(".parquet")]
    if not parquets:
        continue

    folder_name = os.path.relpath(root, base)
    try:
        dataset = ds.dataset(root, format="parquet")
        if "game_id" in dataset.schema.names:
            ids = dataset.to_table(columns=["game_id"]).column("game_id").unique()
            print(f"{folder_name}: {len(ids)} unique games ({len(parquets)} files)")
            all_ids[folder_name] = ids
    except Exception as e:
        print(f"{folder_name}: ERROR - {e}")

if all_ids:
    arrays = [a.cast(pa.string()) for a in all_ids.values()]
    combined = pa.concat_arrays(arrays).unique()
    print(f"\nTotal unique game_ids across everything: {len(combined)}")

# Print last processed move from each engine
print("\n--- Latest eval samples ---")
for engine_label, engine_glob in [("LC0", "lc0"), ("Stockfish", "stockfish")]:
    moves_dirs = []
    for root, dirs, files in os.walk(base):
        if engine_glob in root and "moves" in os.path.basename(root):
            parquets = [f for f in files if f.endswith(".parquet")]
            if parquets:
                moves_dirs.append(root)

    if not moves_dirs:
        print(f"\n{engine_label}: no move data found")
        continue

    latest_file = None
    latest_mtime = 0
    for d in moves_dirs:
        for f in os.listdir(d):
            if f.endswith(".parquet"):
                fpath = os.path.join(d, f)
                mt = os.path.getmtime(fpath)
                if mt > latest_mtime:
                    latest_mtime = mt
                    latest_file = fpath

    if latest_file:
        try:
            table = ds.dataset(latest_file, format="parquet").to_table()
            last_row = table.slice(table.num_rows - 1, 1).to_pandas()
            print(f"\n{engine_label} (from {os.path.relpath(latest_file, base)}):")
            row = last_row.to_dict(orient="records")[0]
            for k, v in row.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"\n{engine_label}: ERROR reading {latest_file} - {e}")