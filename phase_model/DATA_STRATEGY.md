# MIMO — Data Strategy

Complete specification of all model inputs, ground truth labels, and data flow.

---

## Source Data

Lichess game PGNs → `parallel_processor.py` (LC0 nodes=1 / Stockfish depth-14) → 3 parquet tables:
- `games/` — game-level metadata (IDs, Elos, time control, result)
- `moves/` — one row per position/move played
- `possible_moves/` — one row per candidate move per position

`mimo_dataset.py` reads all three, joins them, computes features + labels, and writes train/val/test `.npz` splits (80/10/10 by game ID to prevent leakage).

---

## Model Inputs

### Board Planes — 47 × 8 × 8

| Planes | Description |
|--------|-------------|
| 0-5 | Current position piece maps (P, N, B, R, Q, K for white) |
| 6-11 | Current position piece maps (P, N, B, R, Q, K for black) |
| 12-23 | History position T-1 (same 12 planes) |
| 24-35 | History position T-2 (same 12 planes) |
| 36-37 | Last move from/to squares |
| 38 | Side to move (all 1s = white, all 0s = black) |
| 39-42 | Castling rights (K, Q, k, q) |
| 43 | En passant target square |
| 44-46 | Fifty-move counter (binary-encoded) |

### Tabular Features — 18-dim (position-level)

| # | Feature | Notes |
|---|---------|-------|
| 1 | time_remaining | Clock time left for STM |
| 2 | white_elo | |
| 3 | black_elo | |
| 4 | elo_diff | STM elo − opponent elo |
| 5 | move_no | |
| 6 | color | 0 = white, 1 = black |
| 7 | eval_stm_before | Engine eval of position, STM perspective |
| 8-10 | wdl_stm_before (w/d/l) | Engine WDL of position, STM perspective |
| 11 | initial_time | Time control base (seconds) |
| 12 | increment | Time control increment (seconds) |
| 13 | prev_capture | Was previous move a capture? |
| 14 | in_check | Is STM in check? |
| 15 | eval_std | Std dev of candidate evals (position complexity) |
| 16 | num_captures_frac | Fraction of candidates that are captures |
| 17 | num_checks_frac | Fraction of candidates that give check |
| 18 | num_candidates_frac | Candidate count / 40 (normalized) |

### Per-Candidate Scalars — 11-dim (per move, up to 40 candidates)

| # | Feature | Notes |
|---|---------|-------|
| 1 | eval_stm | This move's eval, STM perspective |
| 2-4 | wdl_stm (w/d/l) | This move's engine WDL, STM perspective |
| 5 | log_nodes | log(nodes searched by engine) |
| 6 | depth | Engine search depth |
| 7 | move_quality | (this_eval − worst) / (best − worst), normalized 0-1 |
| 8 | piece_type | Encoded piece being moved |
| 9 | is_capture | Binary |
| 10 | is_check | Binary |
| 11 | is_checkmate | Binary |

---

## STM Normalization

**All evaluations and WDL values are normalized to side-to-move (STM) perspective.** Positive eval = good for the player about to move. `wdl_stm[0]` = STM's winning probability. This normalization is applied in the dataset builder so the model never needs to reason about "am I white or black."

---

## Ground Truth Labels

### 1. Move Choice → `actual_idx`
- **What:** Index into the candidate move list matching the UCI move the player actually played.
- **Source:** Lichess PGN move matched against the engine's candidate list.
- **Loss:** Cross-entropy over candidate logits.

### 2. WDL Before / After → `win_prob_before`, `win_prob_after`
- **What:** The **actual game result** (not engine WDL), expressed as a [win, draw, loss] vector.
- **Encoding:** `[1, 0, 0]` if STM won the game, `[0, 1, 0]` if draw, `[0, 0, 1]` if STM lost.
- **"Before":** From the perspective of the player about to move.
- **"After":** From the perspective of the player who moves next (the opponent), so the color flips.
- **Why game result, not engine eval?** The model learns "in positions like this, when this player makes this move, they tend to win/draw/lose the game" — which is behavioral, not just positional. This is exactly what we want for predicting human outcomes.
- **Loss:** KL divergence.

### 3. Mistake → `is_mistake`
- **What:** Binary label — did the player's move significantly drop their expected score?
- **Metric:** Expected score = `W + 0.5×D` (STM perspective), where W/D are engine win/draw probabilities per candidate move.
- **Best move:** Candidate with highest expected score (not necessarily highest eval).
- **Drop:** `best_expected_score − played_expected_score`
- **Elo-adaptive thresholds:**

| Elo Range | Threshold (expected score drop) |
|-----------|-------------------------------|
| < 1500 | > 0.20 (20%) |
| 1500–2500 | > 0.15 (15%) |
| > 2500 | > 0.10 (10%) |

- **Outcome shift rule:** If the most likely outcome worsens (W→D, W→L, or D→L) **and** the expected score drop exceeds 5%, always flagged as a mistake regardless of the Elo threshold. This catches moves that change the character of the position even if the raw score drop is modest.
- **Loss:** Binary cross-entropy.
- **Why W+0.5D instead of centipawns?** Centipawn loss is misleading in non-linear positions — a 150cp drop from +8.0 to +6.5 is flagged but irrelevant, while a 50cp drop at +0.3 can flip the game. Expected score is inherently scaled to outcome probability, eliminating the need for fragile Elo-dependent centipawn thresholds.

### 4. Time Spent → `time_spent_log`
- **What:** `log1p(seconds)` — log-transformed actual clock time the player consumed on that move.
- **Source:** Lichess clock annotations in the PGN (`%clk` tags, differenced between consecutive moves).
- **Why log-scale?** Raw time is heavily right-skewed (most moves are fast, some are very long). Log transform makes the distribution more symmetric for regression.
- **Loss:** Huber loss (robust to outliers).

---

## Auxiliary Feedback Loop

Three auxiliary head outputs are computed **before** the move prediction and fed back (detached) as additional context:

```
global_hidden
  ├─→ mistake_head    → mistake_prob (1-dim)  ──┐
  ├─→ time_head       → time_spent  (1-dim)  ──┤ detached (5-dim total)
  ├─→ wdl_before_head → wdl_before  (3-dim)  ──┘
  │                                             │
  ├─→ move_head(global + move_emb + aux_5)    → move_logits
  │
  └─→ wdl_after_head(global + actual_move)    → win_prob_after (stays downstream)
```

- **Detached:** Gradients from the move loss do not flow back through the auxiliary predictions, preventing the move objective from corrupting the auxiliary heads.
- **wdl_after is never fed back:** It depends on knowing which move was played, so it stays downstream of the move prediction.

---

## Data Splits

- **Split method:** By `game_id` (game-level), not by individual positions. Prevents data leakage — no positions from the same game appear in both train and test.
- **Default ratio:** 80% train / 10% validation / 10% test.
- **Shuffling:** DataLoader `shuffle=True` ensures training positions are not sequential through games.

---

## Loss Function

Uncertainty-weighted multi-task loss:

$$\mathcal{L} = \sum_{i} \frac{1}{2\sigma_i^2} \cdot L_i + \log(\sigma_i^2)$$

where $\sigma_i^2$ are learned per-head uncertainty parameters. Heads that are harder to predict automatically receive lower weight, preventing noisy tasks from dominating training.
