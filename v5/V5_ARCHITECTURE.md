# KRUT V5 — Full Architecture Reference

## Overview

`ChessMIMOModelV5` is a single-CNN, multi-head neural network for predicting human chess decisions. One forward pass per position produces **4 outputs** (move selection, mistake probability, time spent, and WDL evaluation) through specialist expert heads that share information via cross-feed fusion.

V5 adds **5 independently toggleable features** on top of the V4 base, each designed for clean ablation:

| Feature | Flag | Default | What It Does |
|---------|------|---------|-------------|
| FiLM Conditioning | `use_film` | **True** | Elo-aware modulation of CNN trunk |
| Phase-Gated Experts | `use_phase_experts` | **True** | Per-phase sub-experts for all 4 heads |
| Contrastive Encoder | `contrastive_embed_dim` | **64** (on) | Preference embeddings via triplet loss |
| Attention Move Head | `move_head_ver` | `'default'` (MLP) | Transformer over legal moves |
| Tactical Enrichment | `use_tactical_enrichment` | **False** | Frozen theme/opening embeddings into trunk |

---

## Input Specification

| Input | Shape | Description |
|-------|-------|-------------|
| `current_planes` | `(B, 23, 8, 8)` | Board representation — planes 0-11: piece planes, 12-15: last two move from/to squares, 16: side-to-move, 17-20: castling rights, 21: en passant, 22: halfmove clock |
| `tabular` | `(B, 20)` | Scalar features — clock_time, w_elo/3000, b_elo/3000, (w_elo-b_elo)/1000, move_no, color, eval_stm, WDL_before (3), initial_time, increment, prev_capture, in_check, eval_std, captures_frac, checks_frac, num_candidates, frac_mistake_moves, frac_excellent_moves |
| `possible_from_sq` | `(B, M)` | Source square indices (0-63) for each candidate move |
| `possible_to_sq` | `(B, M)` | Destination square indices |
| `possible_promo` | `(B, M)` | Promotion piece type (0=none, 1-4=N/B/R/Q) |
| `possible_scalars` | `(B, M, 13)` | Per-move scalar features — includes engine WDL (0-100 scale), is_capture, is_check, piece_type, etc.; slot 12 = is_excellent_move |
| `possible_mask` | `(B, M)` | 1.0 for real moves, 0.0 for padding |
| `actual_idx` | `(B,)` | Index of human's chosen move (training only, optional) |

`M = max_possible` (default 220 = max legal moves in any chess position).

---

## Forward Pass — Full Pipeline

### Stage 1: FiLM Conditioning (optional, on by default)

```
tabular (B, 20) → FiLMConditioner
  → 2-layer MLP (20 → 64 → 64)
  → 6 per-ResBlock projections → (gamma, beta) pairs
  gamma initialized at 1.0, beta at 0.0 (identity init = zero effect before training)
```

**Purpose**: A 1200-Elo and 2500-Elo player see the same board but the CNN learns different feature emphasis for each. FiLM modulates *what features the CNN extracts* based on player context.

~105K params (~3.7% of base model).

### Stage 2: Board Encoder (CNN — runs ONCE per position)

```
current_planes (B, 23, 8, 8)
  → Stem: Conv2d(23→128) + BN + GELU
  → 6 ResBlocks (each: BN→GELU→Conv→BN→GELU→Conv + FiLM modulation + residual)
  → SqueezeExcitation after every 2nd block (3 total)
  → AdaptiveAvgPool2d(1) → board_emb (B, 128)
  → Also retains feature_map (B, 128, 8, 8) for MoveEncoder
```

The CNN runs exactly **once** per position. V1/V2 ran it once per candidate move (B×M forwards) — V3+ eliminated ~54× redundant CNN compute.

### Stage 3: Tabular Encoder

```
tabular (B, 20) → 3-layer MLP (20→64→64→64) with LayerNorm + GELU → tab_emb (B, 64)
```

### Stage 4: Move Encoder (lightweight, per-move)

```
For each candidate move m:
  - Index feature_map at from_sq and to_sq → (B, M, 128) each
  - Learned square embeddings: from_embed(64→32) + to_embed(64→32)
  - Promotion embedding: promo_embed(5→8)
  - Scalar MLP: possible_scalars (13→48→32)
  - Concatenate all → (B, M, 480) → projection → move_emb (B, M, 256)
  - Masked by possible_mask
```

### Stage 5: Cross-Attention

```
context = [board_emb, tab_emb] → (B, 192) → query_proj → (B, 1, 256)
key/value = move_emb (B, M, 256)
→ MultiheadAttention (4 heads, batch_first)
→ LayerNorm → sigmoid gate → attn_out (B, 256)
```

The position representation attends over all legal moves to build a global move-aware context.

### Stage 6: Trunk Fusion

```
fused = [board_emb(128) + tab_emb(64) + attn_out(256)] = (B, 448)
  → Linear(448→256) + LayerNorm + GELU + Dropout
  → trunk_out (B, 256)
```

### Stage 6.5: Tactical Enrichment (optional, OFF by default)

```
current_planes → frozen TacticalPreprocessor
  → ThemeHead: 62-theme multi-label (256-dim embedding)
  → OpeningHead: ~350-opening classification (128-dim embedding)
  → Confidence gate (learned sigmoid scalar, suppresses quiet positions)
  → gated_emb (B, 384)

[trunk_out(256), gated_emb(384)] → projection (640→384→256) → projected (B, 256)
trunk_out = alpha * projected + (1 - alpha) * trunk_out
  where alpha = sigmoid(learned_param), initialized at sigmoid(-3.0) ≈ 0.05
```

**Key constraints**:
- TacticalPreprocessor CNN backbone is ALWAYS frozen during V5 training
- Only confidence_gate, enrichment projection, and residual_alpha are trainable
- Requires pre-trained TacticalPreprocessor checkpoint (trained on Lichess puzzle data)
- Two construction paths: from file path (first training) or config dict (checkpoint reload)

### Stage 7: Phase Detection (optional, ON by default)

```
trunk_out (B, 256) → PhaseEncoder
  → Linear(256→64) + LayerNorm + GELU + Dropout
  → Linear(64→3) → softmax
  → phase_weights (B, 3)  [opening, middlegame, endgame]
```

The PhaseEncoder learns to detect game phase from trunk features (which already encode piece planes, halfmove clock, material, etc.).

**Training signal**: Heuristic game_phase anchors (NOT ground truth) provide an initial starting signal. The anchoring rules are:
- Opening (0): ply ≤ 20
- Endgame (2): no queens + each side ≤ rook + minor, OR queen(s) with no other non-pawn pieces
- Middlegame (1): everything else

The model learns the real phase boundaries through training — these anchors just bootstrap it.

**Phase encoder is frozen by default during expert finetuning** to prevent cross-expert interference (unfreezing one expert shifts the shared encoder and can degrade others).

### Stage 8: Auxiliary Expert Heads (3 heads)

All 3 auxiliary heads are **phase-gated** when `use_phase_experts=True`: each has 3 sub-experts (one per phase), blended by `phase_weights`. Their hidden states feed into the move head via cross-feed fusion.

#### 8a. Mistake Head
```
trunk_out (B, 256) → PhaseGatedExpert (3 × ExpertModule)
  Each: 2-layer MLP (256→128→128) → head (128→1)
  → blended by phase_weights → (hidden: B×128, output: B×1)
Output: raw logit (sigmoid applied at eval/inference only)
```

#### 8b. Time Spent Head
```
trunk_out → PhaseGatedExpert (3 × ExpertModule, same arch)
  → (hidden: B×128, output: B×1)
Output: predicted log(time_spent)
Loss: Huber (delta=2.0)
```

#### 8c. WDL Before Head
```
trunk_out → PhaseGatedExpert (3 × ExpertModule)
  Each: 256→128→128 → head (128→3)
  → blended → (hidden: B×128, output: B×3)
Output: softmax → [W, D, L] probabilities (always White's perspective)
Loss: cross-entropy (soft targets)
```

### Stage 9: Contrastive Encoder (optional, ON by default)

Runs on move_emb **before** the move head — its output is concatenated into the move head input.

```
move_emb (B, M, 256) → ContrastiveEncoder
  → 2-layer MLP (256→128→64) with LayerNorm + GELU
  → contrastive_embed (B, M, 64)

trunk_out (B, 256) → contrastive_anchor_proj → contrastive_anchor (B, 64)
```

**Training**: Soft contrastive triplet margin loss, confidence-weighted by top-3 probability spread:
- Anchor: position embedding projected to 64-dim
- Positive: embedding of human's chosen move
- Negative: embedding of model's highest-scored non-chosen move (near-miss, detached)
- **Confidence weight**: `1 - entropy(top3_probs) / log(3)`
  - Decisive position (one dominant move): confidence → 1, full loss applied
  - Ambiguous position (uniform top-3): confidence → 0, loss effectively suppressed
- `per_sample_loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)`
- `contrastive_loss = sum(per_sample_loss * confidence) / sum(confidence)`

This prevents the triplet loss from fighting itself on true 50/50 positions where multiple moves are equally human-playable. The confidence weights are detached (no gradient through the weighting).

**Inference**: ContrastiveEncoder still runs and feeds contrastive_embed into the move head. The triplet loss doesn't apply (no ground truth), but the learned preference embeddings still enrich move representations.

### Stage 10: Cross-Feed Fusion + Move Head (phase-gated)

The move head consumes outputs from **all** upstream stages: auxiliary expert hidden states (detached), move embeddings, and contrastive embeddings.

```
PhaseGatedMoveHead wraps 3 independent move heads (one per phase)
Each move head receives:
  cross_feed = [trunk_out(256) + wdl_h(128, detached) + mistake_h(128, detached) + time_h(128, detached)] = 640
  + move_emb(256)
  + contrastive_embed(64, if enabled)
  = (B, M, 960) total input per move

3 sub-heads each produce (B, M) scores → blended by phase_weights → (B, M)
Masked with -inf for padded moves
```

**This is where phase-gating is most impactful** — opening repertoire vs endgame technique are completely different move-selection problems.

Move head types (selected via `move_head_ver`):
- `default`: 3-layer MLP (in→256→128→1), processes each move independently
- `deep_4L`: 4-layer MLP
- `wide_512`: wider first layer (512)
- `attention`: 2-layer transformer (4 heads) — moves attend to each other before scoring
- `attention_deep`: 3-layer transformer — **recommended for V5**

---

## Loss: MIMOLossV5

5-head Kendall uncertainty weighting (learned inverse-uncertainty scalers, NOT importance weights):

| Head | Loss Function | Notes |
|------|--------------|-------|
| `move_logits` | Cross-entropy (over ~220 classes) | Lowest Kendall weight because highest raw loss |
| `mistake_prob` | Binary cross-entropy (masked) | Only on valid moves (actual_idx ≥ 0) |
| `win_prob_before` | Soft cross-entropy | Targets are soft WDL distributions |
| `time_spent` | Huber (δ=2.0) | Predicts log-scale time |
| `contrastive` | Triplet margin | Only when contrastive is enabled |

```
total = Σ (exp(-log_var_i) * loss_i + log_var_i)
```

---

## Cross-Feed Fusion

Expert hidden states flow into the move head via **detached** concatenation:

```
cross_feed = [trunk_out, wdl_h.detach(), mistake_h.detach(), time_h.detach()]
```

The `.detach()` means move-head gradients don't backprop through expert heads — each expert trains on its own loss. But the move head *sees* what the experts think (e.g., "this position is a likely mistake" or "White is winning") when scoring moves.

---

## Expert & Move Head Registries

Both experts and move heads are registered in dictionaries for hot-swapping:

**Experts**: `default` (2L×128), `deep_4L` (4L×128), `wide_256` (2L×256), `deep_wide` (4L×256), `phase_gated_default/deep/wide`

**Move heads**: `default` (3L MLP), `deep_4L`, `wide_512`, `attention` (2L transformer), `attention_deep` (3L transformer)

`swap_expert()` and `swap_move_head()` allow runtime replacement. Expert swap is blocked when `use_phase_experts=True` (all experts share the PhaseEncoder).

---

## Preset Configurations

Use `ChessMIMOModelV5.from_preset(name)` to instantiate a standard config.
Constructor defaults match the **V5** (default) preset with `attention_deep` move head.

| Preset | MLP | + Attention | CNN | Hidden | Expert | Contrastive |
|--------|-----|-------------|-----|--------|--------|-------------|
| `v5-minimal` | ~3.9M | — | 128ch / 6 blk | 256 | 128 | 64-dim |
| **`v5` (default)** | ~9.7M | **~14.2M** | 192ch / 8 blk | 384 | 160 | 96-dim |
| `v5-large` | ~19.5M | **~29.1M** | 256ch / 10 blk | 512 | 192 | 128-dim |

**Scaling philosophy**: CNN and move head get the bulk of added capacity (feature
extraction + move prediction are the primary bottleneck for top-1 accuracy).
Contrastive encoder and FiLM scale proportionally.  Auxiliary expert heads
(mistake, time, WDL) get modest increases — they're already near-converged at
lower widths.

### Full parameter breakdown (V5 default, attention_deep)

| Component | Params | % |
|-----------|--------|---|
| CNN (BoardEncoder) | 5.4M | 38% |
| Move Head (attention × 3 phase) | 6.2M | 44% |
| Cross-Attention | 0.8M | 6% |
| Aux Experts (mistake+time+WDL) | 0.8M | 6% |
| FiLM + Trunk + Phase | 0.6M | 4% |
| Contrastive + Anchor | 0.13M | 1% |
| Move Encoder + Tabular | 0.2M | 1% |

---

## Ablation Configurations

```python
# Full V5 (default — recommended)
model = ChessMIMOModelV5.from_preset('v5')

# V5-large for Maia comparison target
model = ChessMIMOModelV5.from_preset('v5-large')

# V5-minimal for rapid iteration / ablation
model = ChessMIMOModelV5.from_preset('v5-minimal')

# Explicit constructor (equivalent to 'v5' preset)
ChessMIMOModelV5(cnn_channels=192, num_res_blocks=8, hidden_dim=384,
                 expert_hidden=160, move_head_ver='attention_deep',
                 contrastive_embed_dim=96)

# V4 equivalent (all V5 features off)
ChessMIMOModelV5.from_preset('v5-minimal',
                             use_film=False, use_phase_experts=False,
                             contrastive_embed_dim=0, move_head_ver='default')

# MLP move head ablation (any preset)
ChessMIMOModelV5.from_preset('v5', move_head_ver='default')  # ~9.7M

# With tactical enrichment
ChessMIMOModelV5.from_preset('v5',
                             use_tactical_enrichment=True,
                             tactical_preprocessor_path='path/to/trained_preprocessor.pt')
```

---

## File Inventory (V5 Codebase)

| File | Role |
|------|------|
| `chess_mimo_model_v5.py` | Model definition, loss, registries |
| `train_mimo.py` | Training loop — `--use-tactical-enrichment`, `--tactical-preprocessor-path` |
| `validate_mimo.py` | Validation — checkpoint-aware tactical reconstruction |
| `infer_mimo.py` | Single-position inference — `compute_game_phase_from_board`, tactical config |
| `finetune_experts.py` | Expert finetuning — `--freeze-phase-encoder`/`--unfreeze-phase-encoder`, per-head LR flags |
| `finetune_player.py` | Player-specific finetuning — tactical config in model construction |
| `mimo_dataset_polars.py` | Polars-based dataset loader — game_phase always-on, material-based heuristics |
| `mimo_uci_engine.py` | UCI engine wrapper — delegates to `MIMOPredictor` from `infer_mimo.py` |

---

## Key Design Principles

1. **Single CNN pass** — V3's core insight: CNN runs once, MoveEncoder indexes the feature map. Eliminated ~54× redundant computation vs V1/V2.

2. **FiLM modulates what the CNN sees** (Elo-dependent feature extraction), PhaseEncoder modulates **which experts process the result** (phase-dependent head blending). Two orthogonal conditioning axes.

3. **Game phase anchors are heuristic, not ground truth** — they bootstrap the PhaseEncoder with reasonable initial signal. The model learns the actual boundaries through training.

4. **Tactical enrichment is a frozen external signal** — the TacticalPreprocessor never trains during V5; only the confidence gate and residual projection adapt.

5. **Every V5 feature is independently toggleable** — clean ablation by flag, no architectural coupling between features.

6. **WDL is always White's perspective** — `[1,0,0]` = White won, `[0,1,0]` = Draw, `[0,0,1]` = Black won. No STM flips.

7. **Kendall weights are inverse-uncertainty, not importance** — the move head gets the lowest weight because it has the highest raw loss (CE over ~220 classes), not because it matters least.
