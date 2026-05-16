# MIMO Opus — Architecture Deep Dive

## Full System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                              DATA PIPELINE                                      ║
║                                                                                  ║
║  ┌────────────┐   ┌────────────┐   ┌────────────────┐                           ║
║  │  Lichess   │   │  LC0       │   │  Stockfish     │                           ║
║  │  PGN dump  │──▶│  Engine    │──▶│  Engine        │                           ║
║  │  (500K+    │   │  nodes=1   │   │  depth=14      │                           ║
║  │   games)   │   │  WDL+eval  │   │  (future pass) │                           ║
║  └────────────┘   └────────────┘   └────────────────┘                           ║
║         │                │                 │                                     ║
║         ▼                ▼                 ▼                                     ║
║  ┌──────────────────────────────────────────────┐                               ║
║  │            Parquet Files                      │                               ║
║  │  games.parquet    actual_moves.parquet         │                               ║
║  │                   possible_moves.parquet       │                               ║
║  └──────────────────────┬───────────────────────┘                               ║
║                         │                                                        ║
║                         ▼                                                        ║
║  ┌──────────────────────────────────────────────┐                               ║
║  │        mimo_dataset_opus.py                   │                               ║
║  │  • Join tables on (game_id, move_no)          │                               ║
║  │  • Generate 47-plane CNN encodings            │                               ║
║  │  • Compute STM-normalized features            │                               ║
║  │  • Split by game_id → train/val/test .npz     │                               ║
║  │  • Multiprocessing for parallelism            │                               ║
║  └──────────────────────┬───────────────────────┘                               ║
║                         │                                                        ║
╠═════════════════════════╪════════════════════════════════════════════════════════╣
║                         ▼            MODEL                                       ║
║                                                                                  ║
║  ┌───────────────────────────────────────────────────────────────────────────┐   ║
║  │                                                                           │   ║
║  │   INPUTS                                                                  │   ║
║  │   ══════                                                                  │   ║
║  │                                                                           │   ║
║  │   current_planes ──┐    tabular ──┐    possible_planes ──┐  poss_scalars  │   ║
║  │   (B, 47, 8, 8)   │    (B, 14)   │    (B, M, 47, 8, 8) │  (B, M, 11)   │   ║
║  │                    │              │                      │       │        │   ║
║  │                    ▼              ▼                      ▼       ▼        │   ║
║  │            ┌──────────────┐ ┌──────────┐        ┌──────────────────────┐  │   ║
║  │            │  SHARED CNN  │ │ TABULAR  │        │     SHARED CNN       │  │   ║
║  │            │              │ │ ENCODER  │        │    (same weights)    │  │   ║
║  │            │  Stem:       │ │          │        │                      │  │   ║
║  │            │  47→128 conv │ │ 14→64    │        │  Each of M moves     │  │   ║
║  │            │  BN + GELU   │ │ 3-layer  │        │  encoded separately  │  │   ║
║  │            │              │ │ LayerNorm│        │  → (B, M, 128)       │  │   ║
║  │            │  Tower:      │ │ GELU     │        │                      │  │   ║
║  │            │  6× ResBlock │ │ Dropout  │        │  ┌─────────────────┐ │  │   ║
║  │            │  (BN→GELU→  │ │          │        │  │ SCALAR ENCODER  │ │  │   ║
║  │            │   Conv→BN→  │ │ → 64-dim │        │  │ 11→48→32 MLP   │ │  │   ║
║  │            │   GELU→Conv │ │          │        │  │ → (B, M, 32)   │ │  │   ║
║  │            │   + skip)   │ └────┬─────┘        │  └───────┬─────────┘ │  │   ║
║  │            │              │      │              │          │           │  │   ║
║  │            │  SE blocks   │      │              │  Concat: 128 + 32   │  │   ║
║  │            │  after every │      │              │  Project: 160→256    │  │   ║
║  │            │  2 ResBlocks │      │              │  → (B, M, 256)      │  │   ║
║  │            │  (3 total)   │      │              └──────────┬───────────┘  │   ║
║  │            │              │      │                         │              │   ║
║  │            │  GAP → 128   │      │                move_emb │              │   ║
║  │            └──────┬───────┘      │                         │              │   ║
║  │                   │              │                         │              │   ║
║  │           board_emb (128)   tab_emb (64)                  │              │   ║
║  │                   │              │                         │              │   ║
║  │                   └──────┬───────┘                         │              │   ║
║  │                          │                                 │              │   ║
║  │                  context (192)                             │              │   ║
║  │                          │                                 │              │   ║
║  │                          ▼                                 ▼              │   ║
║  │                 ┌────────────────────────────────────────────────┐        │   ║
║  │                 │          CROSS-ATTENTION                       │        │   ║
║  │                 │                                                │        │   ║
║  │                 │  Query:  context (192) → proj → (B, 1, 256)   │        │   ║
║  │                 │  Keys:   move_emb            → (B, M, 256)    │        │   ║
║  │                 │  Values: move_emb            → (B, M, 256)    │        │   ║
║  │                 │                                                │        │   ║
║  │                 │  4 attention heads                             │        │   ║
║  │                 │  key_padding_mask for invalid moves            │        │   ║
║  │                 │  LayerNorm on output                          │        │   ║
║  │                 │                                                │        │   ║
║  │                 │  → attn_out (B, 256)                          │        │   ║
║  │                 └──────────────────────┬─────────────────────────┘        │   ║
║  │                                        │                                  │   ║
║  │                                        ▼                                  │   ║
║  │                 ┌────────────────────────────────┐                        │   ║
║  │                 │        FUSION MLP               │                        │   ║
║  │                 │                                  │                        │   ║
║  │                 │  Concat: board(128) + tab(64)    │                        │   ║
║  │                 │          + attn(256) = 448       │                        │   ║
║  │                 │  (opus_with_phase: + phase(16)   │                        │   ║
║  │                 │                     = 464)       │                        │   ║
║  │                 │                                  │                        │   ║
║  │                 │  Linear 448→256                  │                        │   ║
║  │                 │  LayerNorm + GELU + Dropout(0.2) │                        │   ║
║  │                 │  Linear 256→256                  │                        │   ║
║  │                 │  LayerNorm + GELU                │                        │   ║
║  │                 │                                  │                        │   ║
║  │                 │  → global_hidden (B, 256)        │                        │   ║
║  │                 └───────────────┬──────────────────┘                        │   ║
║  │                                 │                                          │   ║
║  │          ┌──────────┬───────────┼───────────┬──────────┐                   │   ║
║  │          ▼          ▼           ▼           ▼          ▼                   │   ║
║  │   ┌───────────┐┌────────┐┌───────────┐┌───────────┐┌────────┐            │   ║
║  │   │ MOVE      ││MISTAKE ││ WDL       ││ WDL       ││ TIME   │            │   ║
║  │   │ LOGITS    ││PROB    ││ BEFORE    ││ AFTER     ││ SPENT  │            │   ║
║  │   │           ││        ││           ││           ││        │            │   ║
║  │   │ Per-move: ││ From   ││ MASKED:   ││ ENRICHED: ││ From   │            │   ║
║  │   │ MLP(glob  ││ global ││ actual    ││ global +  ││ global │            │   ║
║  │   │ ∥ move_   ││ hidden ││ move      ││ actual    ││ hidden │            │   ║
║  │   │ emb)→1   ││        ││ zeroed in ││ move_emb  ││        │            │   ║
║  │   │           ││ →sig   ││ attn mask ││ concat    ││ →1 val │            │   ║
║  │   │ →(B, M)  ││ →(B,1) ││           ││           ││ (log s)│            │   ║
║  │   │ softmax  ││        ││ →(B, 3)   ││ →(B, 3)   ││        │            │   ║
║  │   │ CE loss  ││BCE loss││ KL loss   ││ KL loss   ││ Huber  │            │   ║
║  │   └───────────┘└────────┘└───────────┘└───────────┘└────────┘            │   ║
║  │                                                                           │   ║
║  │   Loss: Σ (1/2σ²ᵢ) × Lᵢ + log(σ²ᵢ)    [learnable uncertainty weights]  │   ║
║  │                                                                           │   ║
║  └───────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                            TRAINING                                              ║
║                                                                                  ║
║  Optimizer:   AdamW (lr=3e-4, weight_decay=1e-4)                                ║
║  Schedule:    Linear warmup (5% of steps) → Cosine decay                        ║
║  Precision:   AMP (float16 forward, float32 gradients) on RTX 4090              ║
║  Grad clip:   max_norm=1.0                                                       ║
║  Batch size:  64 (fits ~8GB VRAM with AMP)                                      ║
║  Epochs:      20-30 (watch val loss plateau)                                    ║
║  Splits:      By game_id (no same-game leakage across train/val/test)           ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                            VALIDATION                                            ║
║                                                                                  ║
║  Move:      Top-1/3/5 accuracy, perplexity, breakdown by game phase             ║
║  Mistake:   AUC-ROC, AUC-PR, accuracy, breakdown by Elo bucket                 ║
║  WDL:       Brier score, ECE (calibration error), accuracy                      ║
║  Time:      MAE, RMSE, Pearson r, Spearman ρ, bucket calibration               ║
║  Leakage:   win_prob_before accuracy < win_prob_after accuracy (MUST hold)      ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Training Pipeline Flow

```
Step 1: Data Generation (your Windows machine)
═══════════════════════════════════════════════
parallel_processor.py → Parquet files
  • LC0 nodes=1 pass: ~10.7 games/sec, ~13h for 500K
  • Stockfish depth=14 pass: separate run afterward
  • Output: games/ actual_moves/ possible_moves/ partitioned by engine

Step 2: Dataset Building (CPU-heavy, ~30min for 500K games)
════════════════════════════════════════════════════════════
python mimo_dataset_opus.py \
    --moves actual_moves/ --games games/ --possible possible_moves/ \
    --output-dir dataset/ --workers 8 --min-elo 0 --max-elo 0

  • Reads all 3 Parquet tables
  • Joins on (game_id, move_no)
  • For EACH position:
    ├── Generate 47-plane encoding of current board
    ├── Generate 47-plane encoding for EACH candidate move's result
    ├── Compute 14 tabular features (STM-normalized)
    ├── Compute 11 scalar features per candidate (STM-normalized)
    ├── Classify game phase (opus_with_phase only)
    └── Determine actual_idx, is_mistake, WDL targets, time_spent target
  • Split by game_id → train.npz, val.npz, test.npz

Step 3: Training (GPU, ~2-4h on RTX 4090 for 500K games)
═════════════════════════════════════════════════════════
python train_mimo_opus.py \
    --train-data dataset/train.npz --val-data dataset/val.npz \
    --output-dir checkpoints/ --epochs 30 --batch-size 64

  • Each batch: single forward pass handles all 5 heads + masking
  • Uncertainty-weighted loss auto-balances tasks
  • Checkpoints: best.pt, latest.pt, periodic epoch saves
  • TensorBoard logging for loss curves + task weights

Step 4: Validation (GPU, ~5min)
═══════════════════════════════
python validate_mimo_opus.py \
    --checkpoint checkpoints/best.pt --data dataset/test.npz

  • Comprehensive metrics for all 5 heads
  • Leakage detection (critical safety check)
  • Calibration curves
  • Breakdown by Elo bucket and game phase
```

---

## Information Flow & Masking Diagram

```
                    WHAT EACH HEAD SEES
                    ════════════════════

                  Position  Candidates  Tabular  Actual   Actual
                  Planes    Planes+Eval Features Move Idx Move Emb
                  ────────  ──────────  ───────  ──────── ────────
move_logits:        ✓          ✓           ✓       ✗*        ✗
mistake_prob:       ✓          ✓           ✓       ✗         ✗
win_prob_before:    ✓       ✓ (MASKED)     ✓       ✗         ✗
win_prob_after:     ✓          ✓           ✓       ✓         ✓
time_spent:         ✓          ✓           ✓       ✗         ✗

  * move_logits SCORES each candidate but doesn't know which was played.
    It ranks them — the loss compares its ranking to actual_idx.

  MASKED = actual move zeroed in cross-attention key_padding_mask
  ✓ (Actual Move Emb) = global_hidden concatenated with actual move's
                         embedding from the candidate pool
```

---

## Things to Check During Training

### 🔴 Red Flags (stop and investigate)

1. **win_prob_before accuracy > 0.85** — masking is broken, model sees the answer
2. **win_prob_before accuracy > win_prob_after accuracy** — impossible if masking works
3. **time_spent loss decreasing but MAE not improving** — possible log-scale mismatch
4. **One task weight → 0** — that head is being ignored; check loss function
5. **move_top1 > 0.60 in epoch 1** — model may be memorizing (check if train/val split leaked)
6. **Training loss drops but val loss doesn't** — overfitting, increase dropout or reduce model

### 🟡 Things to Monitor

1. **Task weight ratios** — watch `w_move_logits` vs others in TensorBoard. If move dominates (>10× others), consider fixed minimum weights
2. **move_top1 by game phase** — opening accuracy should be highest (book moves are predictable), endgame often lower
3. **mistake_prob class balance** — if <5% are mistakes, consider focal loss or oversampling
4. **time_spent by time control** — bullet/blitz/rapid players behave very differently; check if one dominates
5. **Gradient norms per head** — if one head's gradients are 100× another's, the uncertainty weighting may not be enough

---

## Iteration Ideas

### Alternative Attention Mechanisms

| Mechanism | What it does | When to try | Effort |
|-----------|-------------|-------------|--------|
| **Gated cross-attention** | Learned gate controls how much attention output contributes to fusion | If cross-attention dominates and drowns out board/tabular signal | Low — add `nn.Sigmoid()` gate on attn_out |
| **Multi-query attention** | Fewer key/value heads than query heads | If cross-attention is the bottleneck in training speed | Low — use `nn.MultiheadAttention(kdim=lower)` |
| **Relative position attention** | Attention weights modulated by move-relationship (same piece, adjacent squares) | If move_logits struggles with positionally-similar candidates | Medium — custom attention with bias matrix |
| **Slot attention** | Iterative attention that refines move representations | If model struggles to differentiate between many similar candidates | Medium — 2-3 refinement iterations |
| **Perceiver-style** | Latent bottleneck between cross-attention and heads | If M=40 candidates is too many for attention (unlikely) | Medium |
| **Graph attention (GAT)** | Model board as graph (squares = nodes, attacks/defenses = edges) | If CNN struggles with long-range piece interactions | High — full architecture change |

**My recommendation**: Start with the current architecture. If move_top1 plateaus below 35%, try gated cross-attention first (cheapest experiment). If candidates with similar evals are confused, try relative position attention.

### Alternative CNN Architectures

| Architecture | When to try |
|-------------|-------------|
| **Deeper tower (10-12 blocks)** | If validation loss is still decreasing at end of training with 6 blocks |
| **Wider tower (192 or 256 channels)** | If you move to >1M games and have capacity headroom |
| **Spatial attention (CBAM)** | If SE blocks aren't enough — adds spatial attention alongside channel attention |
| **Vision Transformer (ViT)** | If you want to replace CNN entirely — patch the 8×8 board into 4×4 or 2×2 patches |
| **Dual-path** | Separate CNNs for "piece placement" planes (0-35) and "context" planes (36-46) |

### Additional Data Sources

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA ENRICHMENT OPTIONS                          │
│                                                                     │
│  ┌──────────────┐   What it adds          Effort   Impact          │
│  │ Opening Book │   Named opening +       Low      Medium          │
│  │ (ECO/Lichess)│   variation as          (lookup) (opening head   │
│  │              │   categorical feature            accuracy ↑)     │
│  └──────────────┘                                                   │
│                                                                     │
│  ┌──────────────┐   7-piece DTZ/WDL       Medium   High for        │
│  │ Endgame      │   for positions with    (EGTB    endgame head    │
│  │ Tablebases   │   ≤7 pieces; perfect    lookup)  (perfect WDL    │
│  │ (Syzygy)     │   WDL replaces engine            replaces noisy  │
│  │              │   estimate                       engine WDL)     │
│  └──────────────┘                                                   │
│                                                                     │
│  ┌──────────────┐   Material balance,     Low      Medium          │
│  │ Positional   │   king safety score,    (compute (captures       │
│  │ Heuristics   │   pawn structure hash,  from     structural      │
│  │              │   piece mobility count  FEN)     patterns)       │
│  └──────────────┘                                                   │
│                                                                     │
│  ┌──────────────┐   Player's historical   Medium   High for        │
│  │ Player       │   opening repertoire,   (pre-    move prediction │
│  │ History      │   time usage patterns,  compute  — personalizes  │
│  │              │   common mistakes       lookup)  to individual)  │
│  └──────────────┘                                                   │
│                                                                     │
│  ┌──────────────┐   Second engine's eval  High     High            │
│  │ Multi-Engine │   (SF + LC0 together)   (2× run  (LC0 and SF    │
│  │ Consensus    │   as separate features  time)    "see" different │
│  │              │   per candidate move             things)         │
│  └──────────────┘                                                   │
│                                                                     │
│  ┌──────────────┐   "How sharp is this    Low      Medium          │
│  │ Position     │   position?" — std dev  (compute (helps time     │
│  │ Complexity   │   of candidate evals,   from     and mistake     │
│  │ Metrics      │   # of captures/checks  existing prediction)    │
│  │              │   available, mobility   data)                    │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Priority order for enrichment:**
1. **Position complexity metrics** — free to compute from existing data (std dev of candidate evals, count of captures/checks available). Add as tabular features.
2. **Opening book** — ECO code is already in the games table. Could add as a learned embedding (like game phase).
3. **Endgame tablebases** — huge impact for endgame accuracy but requires Syzygy tables (~150GB for 7-piece).
4. **Multi-engine consensus** — comes naturally when the Stockfish pass completes. Add SF eval as a second feature alongside LC0 eval per candidate.
5. **Player history** — the holy grail for personalization but requires per-player modeling (embedding or conditioning).

### Model Architecture Variants to Try

```
Experiment                          Expected gain     Risk
─────────────────────────────────   ─────────────     ────
A. Feedback connections             +1-2% move acc    May overfit to training patterns
   (mistake/time feat → move head)
   
B. Auxiliary "opening classifier"   +opening acc      May not help mid/endgame
   head (predict ECO code)
   
C. Temporal modeling                +2-3% move acc    Much more complex; needs
   (transformer over move sequence  +time accuracy    sequential training batches
   instead of single position)
   
D. Mixture of Experts              +across board      Hard to train, collapse risk
   (route low-Elo vs high-Elo
   through different expert heads)
   
E. Contrastive pre-training        +feature quality   Requires pre-training step
   (learn "which positions are
   similar" before MIMO fine-tune)
```

---

## Feature Summary (Final Architecture)

### Tabular Input (18-dim, STM-normalized)

| # | Feature | Norm | Source |
|---|---------|------|--------|
| 0 | time_remaining | ÷ 3600 | actual_moves |
| 1 | white_elo | ÷ 3000 | games |
| 2 | black_elo | ÷ 3000 | games |
| 3 | elo_diff (white − black) | ÷ 1000 | games |
| 4 | move_no | ÷ 200 | actual_moves |
| 5 | color | 0=Black, 1=White | actual_moves |
| 6 | eval_before (STM) | ÷ 1000 | actual_moves (flipped for Black) |
| 7 | stm_win_before | raw 0-1 | actual_moves (swapped for Black) |
| 8 | draw_perc_before | raw 0-1 | actual_moves |
| 9 | stm_loss_before | raw 0-1 | actual_moves (swapped for Black) |
| 10 | initial_time | ÷ 3600 | games.time_control (parsed) |
| 11 | increment | ÷ 60 | games.time_control (parsed) |
| 12 | previous_move_was_capture | 0\|1 | game_to_position replay |
| 13 | in_check | 0\|1 | chess.Board(fen_before).is_check() |
| 14 | eval_std | ÷ 1000 | std dev of STM evals across candidates |
| 15 | num_captures | fraction | captures / num_candidates |
| 16 | num_checks | fraction | checks / num_candidates |
| 17 | num_candidates | fraction | legal_moves / max_possible |

### Per-Candidate Scalars (11-dim, STM-normalized)

| # | Feature | Norm | Source |
|---|---------|------|--------|
| 0 | eval (STM) | ÷ 1000 | possible_moves (flipped for Black) |
| 1 | stm_win_perc | raw 0-1 | possible_moves (swapped for Black) |
| 2 | draw_perc | raw 0-1 | possible_moves |
| 3 | stm_loss_perc | raw 0-1 | possible_moves (swapped for Black) |
| 4 | log₁₊(nodes) | ÷ 20 | possible_moves |
| 5 | depth | ÷ 40 | possible_moves |
| 6 | move_quality | 0-1 | computed: (eval−worst)/(best−worst) STM |
| 7 | piece_type | 1/6…1.0 | possible_moves.piece |
| 8 | is_capture | 0\|1 | board.piece_at(to_sq) or en passant |
| 9 | is_check | 0\|1 | board_after.is_check() |
| 10 | is_checkmate | 0\|1 | board_after.is_checkmate() |

### Per-Candidate Planes (47 × 8 × 8)

Position AFTER each candidate move, encoded with 2-move history:
- Planes 0-11: piece positions (current)
- Planes 12-23: piece positions (t-1, i.e., before candidate move)
- Planes 24-35: piece positions (t-2)
- Planes 36-39: last move from/to squares (t-1, t-2)
- Plane 40: side to move
- Planes 41-44: castling rights
- Plane 45: en passant square
- Plane 46: fifty-move counter / 100

### Current Position Planes (47 × 8 × 8)

Same encoding as above, but for the position BEFORE any move is played.
