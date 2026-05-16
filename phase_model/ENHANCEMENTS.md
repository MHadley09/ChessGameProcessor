# Potential Enhancements

Future additions to MIMO, organized by when they make sense to add.

---

## After Baseline Training (v1 results in hand)

### Eval Change Magnitude Head
**What:** Predict the continuous eval change (centipawn loss) caused by the player's move, as a separate output head.
**Why:** The current mistake head is binary (blunder or not). A continuous eval-change head captures the full spectrum — brilliant sacrifices (positive change), slight inaccuracies (small negative), and catastrophic blunders (large negative). Gives the model a richer gradient signal about move quality.
**When:** After baseline training, if the binary mistake head shows weak gradients or the move predictor doesn't differentiate well between "slightly suboptimal" and "terrible" moves.

### Game Phase Embedding (with_phase)
**What:** Already built — `nn.Embedding(3, 16)` for opening/middlegame/endgame.
**Why:** Players behave differently by phase: opening = theory recall, middlegame = calculation, endgame = technique. Phase context helps all heads.
**When:** Immediate A/B test after baseline. Code is ready in `with_phase/`.

---

## After Player-Specific Fine-Tuning Pipeline Exists

### Move Familiarity
**What:** Encode how familiar a specific player is with the current position type, based on their game history.
**Why:** Players are faster, more accurate, and less prone to mistakes in structures they've seen before. A player with 200 Najdorf games will play mainline moves instantly; the same player in an unfamiliar Grünfeld will think longer and err more.
**Representation:** Likely a small game-history encoder (not a simple scalar) — e.g., embedding similarity between current board features and a learned summary of the player's past positions.
**When:** Only useful during fine-tuning on a specific player's history (~500-1000 games). The base model trains on anonymous Lichess games where individual history isn't available. Design this once baseline results reveal what the model struggles with on personalized predictions.

---

## After Opening Book / Tablebase Integration

### Opening Book Probability
**What:** Binary or continuous signal — is the current position still within known opening theory?
**Why:** In-theory positions have near-deterministic move choices for strong players. The model should learn to be highly confident in book positions and more uncertain outside them. Also improves time prediction (book moves are instant).
**Data source:** Lichess opening explorer API or a local polyglot book file.
**When:** After baseline is solid. Requires an external data source not currently in the pipeline.

### Endgame Tablebase Features
**What:** For positions with ≤7 pieces, inject the tablebase-proven outcome (win/draw/loss with DTZ).
**Why:** Perfect ground truth in simplified positions. The model can learn that strong players play accurately in tablebase positions and weaker players don't.
**Data source:** Syzygy 7-piece tablebases (requires ~140GB on disk).
**When:** After opening book integration. Lower priority since endgame positions are a small fraction of training data.

---

## Speculative / Research-Grade

### Multi-Engine Consensus
**What:** Use agreement/disagreement between LC0 and Stockfish evaluations as a position complexity feature.
**Why:** When engines disagree, positions are genuinely complex — humans are more likely to err. When they agree, the "correct" move is clearer.
**Prerequisite:** Stockfish pass completed on the same dataset.
**When:** After both LC0 and SF passes are done and baseline is trained.

### Relative Position Attention
**What:** Attention mechanism that encodes spatial relationships between pieces on the board, rather than treating the 8×8 grid as a flat image.
**Why:** Chess has inherent relational structure (piece attacks, defenses, pins) that CNNs learn implicitly but attention could capture explicitly.
**When:** Research iteration — only if CNN-based baseline plateaus and you need architectural improvements.

### Player Style Embedding
**What:** Learnable per-player embedding vector that captures playing style (aggressive, positional, tactical, etc.).
**Why:** Different players make systematically different choices in identical positions. A style embedding lets the model condition predictions on who is playing.
**When:** Requires a dataset with player IDs and enough games per player to learn meaningful embeddings. Fine-tuning phase.
