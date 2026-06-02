from pathlib import Path
base = Path('output/parquet/lc0/791556')
w = sorted([d for d in base.iterdir() if d.is_dir()])[0]
for f in sorted(w.rglob('*'))[:20]:
    print(f.relative_to(w))
