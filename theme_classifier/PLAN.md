# Lichess Puzzle Theme & Opening Classifier — Design Plan

## Goal

Build preprocessing models (theme classifier + opening classifier) trained on
~6M Lichess puzzles that produce **tactical embeddings** and **opening embeddings**
to inject into V5's trunk, enriching its position representation before the
expert heads and move head run.

## Data Source

**Lichess Puzzle CSV** (`lichess_db_puzzle.csv.zst`, ~200MB compressed, ~6M rows):

```
PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
```

Key fields:
- `FEN`: position *before* the opponent's setup move
- `Moves`: UCI moves — moves[0] is the setup move; moves[1:] is the solution
- `Themes`: space-separated multi-label tags (~62 unique themes)
- `GameUrl`: `https://lichess.org/{gameId}/{color}#{ply}` — source game
- `OpeningTags`: underscore-joined opening names (only set for move < 20)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    PUZZLE DATA PIPELINE                  │
│                                                         │
│  CSV → Game Fetch (Lichess API) → Replay to Position    │
│      → LC0 All-Legal-Moves → NPZ Shards               │
│                                                         │
│  Output: V5-compatible shards + themes[] + openings[]   │
└─────────────────────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌─────────────────────┐  ┌─────────────────────┐
│  Theme Classifier   │  │ Opening Classifier  │
│  (multi-label)      │  │ (multi-class)       │
│  62 sigmoid outputs │  │ ~350 softmax outputs│
│  BCE loss           │  │ CE loss             │
│                     │  │                     │
│  Shared CNN trunk   │  │  Shared CNN trunk   │
│  (23 input planes)  │  │  (23 input planes)  │
│  tactical_emb: 128d │  │  opening_emb: 64d   │
└─────────┬───────────┘  └──────────┬──────────┘
          │                         │
          └────────────┬────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    V5 INTEGRATION                       │
│                                                         │
│  board_planes → V5_CNN → trunk_out (existing)          │
│  board_planes → frozen ThemeClassifier → tactical_emb   │
│  board_planes → frozen OpeningClassifier → opening_emb  │
│                                                         │
│  enriched = cat(trunk_out, tactical_emb, opening_emb)  │
│  → Expert heads, Move head operate on enriched trunk    │
└─────────────────────────────────────────────────────────┘
```

## Data Processor Design

### Phase 1: Parse CSV + Extract Game IDs
- Read `lichess_db_puzzle.csv` (or .zst)
- Extract unique game IDs from GameUrl
- ~6M puzzles → ~3-4M unique games (puzzles share source games)

### Phase 2: Fetch Games from Lichess API
- `POST https://lichess.org/games/export/_ids` — up to 300 IDs per request
- Rate limit: 15 games/sec anonymous, 25/sec with API token
- Response: PGN stream with all requested games
- Cache fetched PGNs to SQLite (keyed by game ID) for resumability
- Estimated time: ~44-74 hours for full dataset (subset first)

### Phase 3: Replay + Extract Position
For each puzzle:
1. Look up cached PGN by game ID
2. Replay game to the puzzle ply number (from URL's `#ply`)
3. Verify FEN matches (apply moves[0] and compare)
4. The **actual move** = what was played in the game at the puzzle position
   (usually == moves[1] from CSV, but verified from the actual game PGN)
5. Truncate PGN to puzzle position (no future moves)
6. Extract: move history (for game_to_position), player Elos, time control

### Phase 4: LC0 Evaluation
- Reuse SyncDirectEvaluator / SyncBatchEvaluator from existing pipeline
- evaluate_all_legal_moves() at the puzzle position
- Same output format: move_uci, score_cp, wdl, nodes, depth, etc.

### Phase 5: Write NPZ Shards
V5-compatible fields:
- `current_planes` (23, 8, 8) — board_to_planes
- `possible_scalars` (M, 13) — per-move LC0 evals
- `possible_mask` (M,) — legal move mask
- `possible_uci` (M,) — UCI strings
- `tabular` (20,) — game-level features (Elo, time, etc.)
- `actual_idx` — index of actual move played
- `is_mistake` — computed from eval drop
- `win_prob_before` (3,) — WDL
- `time_spent_log` — time taken
- `fen_before` — FEN string
- `game_to_position` — UCI move history

Additional fields:
- `themes` (62,) — multi-hot binary vector
- `opening_idx` — integer label for opening classification
- `opening_tags` — raw opening tag string
- `puzzle_rating` — puzzle difficulty rating
- `puzzle_id` — Lichess puzzle ID

## Theme Classifier

### Architecture
```python
class ThemeClassifier(nn.Module):
    # CNN backbone: same 23 input planes as V5
    # ResNet-style blocks (shared with opening classifier optionally)
    # Global average pool → trunk_out
    # FC layers → tactical_embedding (256-dim)    ← THIS gets injected
    # FC → 62 theme logits (sigmoid)
```

### Training
- Loss: BCEWithLogitsLoss (multi-label)
- Metrics: per-theme F1, macro F1, exact match ratio
- Label smoothing for rare themes
- Class weighting (fork: 829K, bodenMate: 4K — huge imbalance)
- Train/val/test: 80/10/10 split by puzzle ID
- ~6M examples, batch 512, should converge in 5-10 epochs

### Theme Label Space (62 tags)
Tactical motifs: fork, pin, discoveredAttack, skewer, doubleCheck,
  xRayAttack, interference, deflection, attraction, clearance,
  intermezzo, sacrifice, capturingDefender, trappedPiece, hangingPiece,
  exposedKing, zugzwang, quietMove, defensiveMove

Mate patterns: mate, mateIn1, mateIn2, mateIn3, mateIn4, mateIn5,
  backRankMate, smotheredMate, bodenMate, anastasiaMate,
  doubleBishopMate, arabianMate, hookMate, killBoxMate,
  vukovicMate, dovetailMate

Phases: opening, middlegame, endgame, pawnEndgame, rookEndgame,
  bishopEndgame, knightEndgame, queenEndgame, queenRookEndgame

Special: advancedPawn, promotion, underPromotion, enPassant,
  attackingF2F7, kingsideAttack, queensideAttack, castling

Length: oneMove, short, long, veryLong

Meta: crushing, advantage, equality, master, superGM, masterVsMaster

## Opening Classifier

### Architecture
```python
class OpeningClassifier(nn.Module):
    # CNN backbone (same or separate from theme classifier)
    # Global average pool → trunk_out
    # FC layers → opening_embedding (128-dim)      ← THIS gets injected
    # FC → ~350 opening class logits (softmax)
```

### Training
- Loss: CrossEntropyLoss (multi-class, single label)
- Only train on puzzles with OpeningTags set (move < 20)
- ~2-3M examples with opening labels
- Top-1 and Top-5 accuracy metrics
- This may be replaced with deterministic opening book lookup later

## V5 Integration Plan (Option A: Trunk Enrichment + Confidence Gate)

### Injection Point
After V5's CNN trunk, before expert heads:

```python
# In V5 forward():
trunk_out = self.cnn(board_planes)                    # (B, trunk_dim)

# Frozen classifiers produce embeddings
tactical_emb = self.frozen_theme.get_embedding(planes) # (B, 256)
opening_emb = self.frozen_opening.get_embedding(planes) # (B, 128)
combined = cat(tactical_emb, opening_emb)               # (B, 384)

# Confidence gate (trainable) suppresses on quiet positions
confidence = self.confidence_gate(combined)              # (B, 1) sigmoid
gated_emb = combined * confidence                        # (B, 384)

# Project back to trunk_dim
enriched = cat(trunk_out, gated_emb)                     # (B, trunk_dim + 384)
projected = self.enrichment_proj(enriched)               # (B, trunk_dim)

# Residual blend (learnable alpha, starts at 0 = pure trunk)
alpha = sigmoid(self.residual_alpha)
blended = alpha * projected + (1 - alpha) * trunk_out

# Expert heads and move head operate on blended trunk
```

### Changes Required to V5
1. Add `TacticalPreprocessor` as a frozen submodule
2. Add `enrichment_proj` (Linear → Mish → Linear, ~250K params)
3. Add `confidence_gate` (Linear → Mish → Linear → Sigmoid, ~25K params)
4. Add `residual_alpha` (1 scalar param, initialized to 0)
5. Expert heads UNCHANGED — they still receive trunk_dim input
6. Move head UNCHANGED — same interface
7. FiLM UNCHANGED — operates on trunk as before (or optionally on
   tactical embedding via TacticalFiLMAdapter for Elo-dependent tactics)

### Fine-tuning Strategy
1. Train theme + opening classifiers on puzzle data (separate step)
2. Freeze classifier weights
3. Load pretrained V5 weights (unchanged)
4. Initialize projection, gate, alpha from scratch
5. Fine-tune: gate + projection + alpha + V5 heads on existing training data
6. The classifier CNNs add inference cost but zero training cost

## Distribution Mismatch Mitigation

Puzzle positions are tactically loaded. The confidence gate is the primary
mitigation:
- Gate is a learned sigmoid scalar that scales the 384-dim embedding
- Trained on puzzle data, it learns high confidence for tactical positions
- During V5 fine-tuning on mixed data, gate learns to suppress on quiet
  positions → V5's trunk dominates for calm positions
- The residual_alpha parameter starts at 0 (pure trunk) and gradually
  learns how much tactical enrichment helps
- Additional augmentation: inject quiet positions from V5 dataset with
  all-zero theme labels during classifier training (optional, may help
  the gate learn faster)

## File Manifest

```
theme_classifier/
├── PLAN.md                         # This document
├── puzzle_dataset_processor.py     # CSV → NPZ pipeline
├── puzzle_dataset.py               # PyTorch Dataset for training
├── tactical_models.py              # ThemeClassifier + OpeningClassifier
├── train_theme.py                  # Theme classifier training
├── train_opening.py                # Opening classifier training
├── infer_theme.py                  # Theme inference (single position)
├── infer_opening.py                # Opening inference (single position)
├── evaluate_classifiers.py         # Evaluation metrics
└── integrate_v5.py                 # V5 enrichment wrapper
```

## Estimated Timeline
- Data processor: 1 day coding + variable fetch time
- Theme classifier training: ~2-4 hours on RTX 4090
- Opening classifier training: ~1-2 hours
- V5 integration + fine-tuning: 1 day
- Evaluation: 0.5 day

## Open Questions for Michael
1. Start with subset (e.g., 500K highest-rated puzzles) or go full 6M?
2. Do you have a Lichess API token for faster game fetching?
3. Theme classifier embedding dim: 128 or 256?
4. Should theme + opening share a CNN trunk or be fully separate?
5. For V5 integration: concatenate embeddings to trunk_out, or use a
   separate cross-attention mechanism?
