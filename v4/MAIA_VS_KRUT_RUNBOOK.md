# Maia-2 vs KRUT Evaluation Runbook

## What This Compares

**KRUT (MIMO V4)**: Multi-head architecture trained on Lichess games. Predicts
moves via softmax over legal-move logits.

**Maia-2**: LC0 network weights (no-search, pure policy net). Run through the
LC0 engine with `nodes=1`. Different weight files target different Elo ranges.
Maia-2 is **not** a Python pip package — it's LC0 weights.

## Metrics

| Metric | Description |
|--------|-------------|
| Top-1 | Fraction where the model's #1 pick matches the human move |
| Top-2 | Human move is in the model's top 2 |
| Top-3 | Human move is in the model's top 3 |
| Top-5 | Human move is in the model's top 5 |
| Log-loss | `-log(P(actual_move))` averaged. Penalises confident wrong predictions heavily. |
| n | Number of positions in the bucket |

## Bucketing

Positions are bucketed by the **side-to-move Elo** in 100-point ranges
(e.g. 1500-1599). In a single game, White's moves go to White's Elo bucket
and Black's moves go to Black's Elo bucket.

## Prerequisites

1. **LC0 binary** — build or download from https://lczero.org
2. **Maia-2 weights** — `.pb.gz` network file for the target Elo range
3. **KRUT checkpoint** — `best.pt` or `best_v4.pt` from training
4. **Test data**:
   - NPZ shards for KRUT (`MIMOCompactDataset`)
   - Original parquet files for Maia-2 (because NPZ shards don't store FENs)

## Usage

### Full comparison (KRUT + Maia-2)

```bash
python maia_vs_krut_eval.py \
    --krut-checkpoint checkpoints/best_v4.pt \
    --data-dir dataset/v1/test \
    --maia-weights maia2-rapid.pb.gz \
    --lc0-path ./lc0 \
    --parquet-dir output/parquet/lc0/test \
    --output-dir comparison_results \
    --batch-size 512
```

### KRUT only

```bash
python maia_vs_krut_eval.py \
    --krut-checkpoint checkpoints/best_v4.pt \
    --data-dir dataset/v1/test \
    --skip-maia \
    --output-dir krut_results
```

### Quick test (subsample)

```bash
python maia_vs_krut_eval.py \
    --krut-checkpoint checkpoints/best_v4.pt \
    --data-dir dataset/v1/test \
    --maia-weights maia2-rapid.pb.gz \
    --lc0-path ./lc0 \
    --parquet-dir output/parquet/lc0/test \
    --max-positions 10000 \
    --output-dir comparison_quick
```

### Using a flat FEN CSV instead of parquet

```bash
python maia_vs_krut_eval.py \
    --krut-checkpoint checkpoints/best_v4.pt \
    --data-dir dataset/v1/test \
    --maia-weights maia2-rapid.pb.gz \
    --lc0-path ./lc0 \
    --fen-csv test_positions.csv \
    --output-dir comparison_results
```

CSV format: `fen,actual_move,stm_elo` (header row required).

## Output Files

| File | Contents |
|------|----------|
| `comparison_metrics.json` | All metrics, machine-readable |
| `comparison_report.txt` | Side-by-side text table |

## Notes

- Maia-2 inference is position-by-position via UCI (slower than KRUT's batched
  DataLoader). For the full test set, expect ~100-500 positions/sec depending
  on GPU and backend.
- The `--lc0-backend` flag defaults to `cuda-fp16`. Use `cudnn-fp16` or `cpu`
  if needed.
- KRUT positions and Maia positions don't need to be 1:1 aligned. The script
  computes aggregate accuracy per bucket independently. But if they come from
  the same test split in the same order, the comparison is maximally fair.
