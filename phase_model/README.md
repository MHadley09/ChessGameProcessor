# MIMO with Phase — Chess Behaviour Prediction Model

**Multi-Input Multi-Output** model predicting human chess behaviour from
board position, candidate moves, game context, **and game phase**.

This is the **with_phase** variant. The only difference from base
`mimo` is the addition of a learned game-phase embedding
(opening / middlegame / endgame) that is concatenated into the context
before cross-attention and fusion.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  current_planes (B,47,8,8)   tabular (B,10)   possible_planes (B,M,47,8,8)
│                              game_phase (B,)    possible_scalars (B,M,6)
└────────┬───────────────────────┬────────────────────┬────────────────┘
         │                       │                    │
  ┌──────▼──────┐      ┌────────▼────────┐   ┌──────▼──────┐
  │  Shared CNN  │      │  Tabular MLP    │   │  Shared CNN  │
  │  6 ResBlocks │      │  3-layer + LN   │   │  (same wts)  │
  │  + 3 SE blks │      │  dropout 0.1    │   │  + scalar    │
  │  GELU activ  │      │  → 64-dim       │   │  MLP (6→32)  │
  │  → 128-dim   │      └────────┬────────┘   │  → 160→256   │
  └──────┬───────┘               │            └──────┬───────┘
         │                       │                   │
         │              ┌────────▼────────┐          │
         │              │ Phase Embedding │          │
         │              │ nn.Emb(3,16)   │          │
         │              └────────┬────────┘          │
         │                       │                   │
         └───────────┬───────────┘                   │
                     │                               │
            ┌────────▼────────┐              ┌───────▼───────┐
            │ Context (208-d) │──── query ──▶│Cross-Attention│
            └────────┬────────┘              │ 4 heads       │
                     │                       └───────┬───────┘
                     │                               │
            ┌────────▼───────────────────────────────▼────────┐
            │         Fusion MLP (464 → 256)                  │
            │         2 layers, LayerNorm, GELU, dropout 0.2  │
            └────────┬────────────────────────────────────────┘
                     │ global_hidden (256-d)
     ┌───────┬───────┼───────┬──────────┐
     ▼       ▼       ▼       ▼          ▼
  ┌──────┐┌──────┐┌──────┐┌──────┐ ┌──────┐
  │move  ││mist- ││WDL   ││WDL   │ │time  │
  │logits││ake   ││before││after │ │spent │
  │(per- ││prob  ││(mask)││(+act)│ │(log) │
  │move) ││      ││      ││emb)  │ │      │
  └──────┘└──────┘└──────┘└──────┘ └──────┘
```

**~5.5M parameters** (right-sized for ≤500K-game datasets on RTX 4090)

### Key CNN Details

- **Stem**: Conv2d 47→128, BN, GELU
- **Tower**: 6 × pre-activation ResBlock (BN→GELU→Conv→BN→GELU→Conv + skip)
- **SE blocks**: Squeeze-and-Excitation after every 2 residual blocks (3 total)
- **Pooling**: Global average pool → 128-dim vector

### Candidate Move Encoding

Each candidate move is encoded as:
- **Plane features**: 47-plane position after the move → shared CNN → 128-dim
- **Scalar features**: eval_stm/1000, stm_win%, draw%, stm_loss%, log₁₊(nodes)/20, depth/40 → MLP → 32-dim
- **Combined**: concatenated (160-dim) → linear projection → 256-dim

---

## 5 Output Heads & Masking Strategy

| # | Head | Output | Masking | Target |
|---|------|--------|---------|--------|
| 1 | `move_logits` | (B, M) per-move scores | Invalid moves → -∞ | Cross-entropy vs actual_idx |
| 2 | `mistake_prob` | (B, 1) sigmoid | No actual_idx info in global_hidden | BCE vs is_mistake |
| 3 | `win_prob_before` | (B, 3) WDL softmax | Actual move **zeroed in cross-attention** key_padding_mask | KL-div vs game result WDL |
| 4 | `win_prob_after` | (B, 3) WDL softmax | Sees actual move embedding (concatenated) | KL-div vs game result WDL |
| 5 | `time_spent` | (B, 1) scalar | time_spent **excluded from tabular** entirely | Huber loss vs log₁₊(seconds) |

### What each head CANNOT see:

- **move_logits**: N/A (must see candidates to rank them)
- **mistake_prob**: which move was played, game result
- **win_prob_before**: which move was played (masked from attention), game result, eval_after
- **win_prob_after**: game result
- **time_spent**: time_spent (not in inputs), game result

---

## Tabular Features (10-dim — no leakage)

All evals and WDL are normalised to **side-to-move (STM) perspective**: positive
eval = good for STM; stm_win = STM's winning probability.

| # | Feature | Normalisation |
|---|---------|---------------|
| 0 | time_remaining | ÷ 3600 |
| 1 | white_elo | ÷ 3000 |
| 2 | black_elo | ÷ 3000 |
| 3 | elo_diff | ÷ 1000 |
| 4 | move_no | ÷ 200 |
| 5 | color | 0=Black, 1=White |
| 6 | eval_stm_before | ÷ 1000 (flipped for Black) |
| 7 | stm_win_before | raw 0-1 (White's win% when White to move, Black's when Black) |
| 8 | draw_perc_before | raw 0-1 |
| 9 | stm_loss_before | raw 0-1 (opponent's win%) |

**Excluded**: `time_spent` (target), `result`/`winner` (target), `eval_after` (leaks for WDL_before), `static_eval_before` (not populated), `mate_count_before` (not populated)

---

## Files

| File | Description |
|------|-------------|
| `chess_mimo_model.py` | Model + loss (PyTorch) |
| `mimo_dataset.py` | Parquet → .npz dataset builder with multiprocessing |
| `train_mimo.py` | Training loop with AMP, warmup-cosine, uncertainty weighting |
| `validate_mimo.py` | Comprehensive 5-head validation + leakage detection |
| `README.md` | This file |

---

## Usage

### 1. Build dataset

```bash
python mimo_dataset.py \
    --moves  data/actual_moves/ \
    --games  data/games/ \
    --possible data/possible_moves/ \
    --output-dir data/dataset \
    --max-possible 40 \
    --min-elo 0 \
    --max-elo 0 \
    --workers 8
```

Produces `train.npz`, `val.npz`, `test.npz` split **by game_id** (no leakage).

Both `--min-elo` and `--max-elo` default to 0 (off). Set to any value 1-9999 to enable filtering.

### 2. Train

```bash
python train_mimo.py \
    --train-data data/dataset/train.npz \
    --val-data   data/dataset/val.npz \
    --output-dir checkpoints/run1 \
    --epochs 30 \
    --batch-size 64 \
    --lr 3e-4 \
    --device cuda
```

### 3. Validate

```bash
python validate_mimo.py \
    --checkpoint checkpoints/run1/best.pt \
    --data       data/dataset/test.npz \
    --output-dir results/run1
```

---

## Differences from Prior Versions

### vs v1 (chess_mimo_model_with_time_spent)

| Aspect | v1 | Current |
|--------|-----|------|
| CNN | 3 conv layers, no residuals | 6 ResBlocks + 3 SE blocks |
| Possible moves | Scalar only (B,218,4) | Full 47-plane CNN + scalar (B,40,6) |
| Move scoring | Global→218 fixed slots | Per-move MLP on individual embeddings |
| Attention | Self-attention on move scalars | Cross-attention: context→move embeddings |
| Tabular leak | time_spent in features (BUG) | time_spent excluded |
| Activations | ReLU throughout | GELU throughout |
| Normalisation | BatchNorm | LayerNorm where possible (MLP, fusion) |
| Loss weighting | Fixed manual weights | Learnable uncertainty weighting |
| Masking | Two forward passes (external) | Single pass, built into model |
| Params | ~1.5M | ~5.5M |

### vs v2 (train_chess_mimo_v2_with_arch_flag)

| Aspect | v2 | Current |
|--------|-----|------|
| time_spent leak | YES (tabular index 1) | NO — excluded from tabular entirely |
| Forward passes | 2 (full + masked) | 1 (masking integrated in model) |
| Feedback variant | Optional via flag | Not needed — cross-attention naturally shares info |
| Data split | Random row split | Game-ID split (prevents same-game leakage) |
| Scheduler | CosineAnnealing | Warmup + Cosine |
| AMP | No | Yes |

### vs v3 (mimo_avocado)

| Aspect | v3 | Current |
|--------|-----|------|
| CNN depth | 3 conv layers | 6 ResBlocks + SE |
| Per-move scalars | None (planes only) | 6-dim (eval, WDL, nodes, depth) |
| Tabular dim | 10 (with 2 wasted slots) | 10 (clean, no padding — includes engine WDL before) |
| win_prob_before masking | Not implemented in training loop | Built into model forward |
| win_prob_after | Same as all other heads | Gets actual-move embedding concatenated |
| Mistake detection | eval_before vs eval_after | Best-candidate vs played-candidate |
| Dataset building | Single-process | Multiprocess with configurable workers |
| Train/val split | Not present (single .npz) | By game_id (train/val/test) |
| Loss | Fixed weights | Uncertainty-weighted (learnable) |

---

## Hyperparameter Recommendations (RTX 4090)

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| batch_size | 64 | ~8GB VRAM with AMP; increase to 128 if you have headroom |
| lr | 3e-4 | AdamW; warmup prevents early instability |
| epochs | 20-30 | Watch val loss plateau |
| max_possible | 40 | Covers 99%+ of positions |
| cnn_channels | 128 | Could try 192 for larger datasets |
| res_blocks | 6 | Sweet spot for this data scale |
| hidden_dim | 256 | Match to CNN output width |
| weight_decay | 1e-4 | Standard for AdamW |
| warmup | 5% of total steps | Stabilises early training |

---

## Schema Compatibility

Built for the Parquet schema in `parquet_schema.py`:

- **games**: 32 fields (game_id, white_elo, black_elo, result, ...)
- **actual_moves**: 31 fields (fen_before, fen_after, time_spent, time_remaining, eval_before, game_to_position, ...)
- **possible_moves**: 22 fields (fen_before, fen_after, eval, WDL%, nodes, depth, move, ...)

All three tables include `evaluated_by` and `evaluator_version` for engine filtering.

Plane encoding: 47 planes from `plane_codec.py` (current + 2-history positions).

---

## Game-Phase Embedding (with_phase only)

### What it is

A learned 16-dimensional embedding that encodes the phase of the game:
- **0 = Opening**: move_no ≤ 12 AND ≥ 12 minor/major pieces on board
- **1 = Middlegame**: everything else
- **2 = Endgame**: ≤ 6 minor/major pieces on board (regardless of move number)

### Why it helps

Humans play very differently across phases:
- **Opening**: Book moves, fast play, pattern matching
- **Middlegame**: Deep calculation, longer thinks, more mistakes at lower Elo
- **Endgame**: Technique-driven, different piece valuations, time pressure effects

A learned embedding lets the model capture these behavioral shifts without
hard-coding assumptions about how phases affect each head.

### How phase is classified

```python
def classify_game_phase(fen: str, move_no: int) -> int:
    piece_part = fen.split()[0]
    minor_major = sum(1 for c in piece_part if c in 'nbrqNBRQ')
    if move_no <= 12 and minor_major >= 12:
        return 0   # opening
    elif minor_major <= 6:
        return 2   # endgame
    else:
        return 1   # middlegame
```

Uses piece count from FEN (available via `fen_before`) rather than just move
number, so it correctly identifies early endgames from exchanges.

### Architecture integration

- `nn.Embedding(3, 16)` → 48 extra parameters (negligible)
- Concatenated with board_emb + tabular_emb before cross-attention query
- Also included in fusion: board (128) + tabular (64) + **phase (16)** + attn (256) = 464
- Backward compatible: if `game_phase=None` is passed, defaults to middlegame (1)

### Difference from base mimo

This is the **only** difference. All other architecture, masking, features,
and training logic are identical to `mimo`.
