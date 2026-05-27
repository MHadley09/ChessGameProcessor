# MIMO — Live Inference Guide

How to use the trained model to predict the next move in a live game.

---

## Assumptions

You have:
- A trained `best.pt` checkpoint
- A running chess engine (LC0 or Stockfish) for candidate evaluation
- The current game state: FEN, move history, clock times, player ratings
- Python with PyTorch, python-chess installed

You do NOT need the training parquet files — everything is computed on the fly.

---

## Inference Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                     LIVE GAME STATE                               │
│                                                                    │
│  • FEN: "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq" │
│  • Move history: ["e2e4", "g8f6"]                                 │
│  • White clock: 285s remaining (started 300+3)                    │
│  • Black clock: 297s remaining                                    │
│  • White Elo: 1850, Black Elo: 1920                               │
│  • White to move, move_no = 3                                     │
│                                                                    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                  STEP 1: Generate Candidate Moves                 │
│                                                                    │
│  board = chess.Board(fen)                                         │
│  legal_moves = list(board.legal_moves)  # typically 20-40 moves   │
│                                                                    │
│  For EACH legal move:                                              │
│    • Push move on board → get fen_after                           │
│    • Run engine analysis (LC0 nodes=1 or SF depth=14):            │
│      → eval, WDL (white_win%, draw%, black_win%), nodes, depth    │
│    • Record: move UCI, from_sq, to_sq, piece, fen_after           │
│    • Pop move (restore board)                                      │
│                                                                    │
│  Also evaluate the CURRENT position (before any move):            │
│    → eval_before, WDL_before                                      │
│                                                                    │
│  ⏱ Latency: ~50ms with LC0 batch, ~2s with SF depth=14           │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                  STEP 2: Build Model Inputs                       │
│                                                                    │
│  from plane_codec import board_to_planes                          │
│                                                                    │
│  A. Current position planes (47, 8, 8)                            │
│     board = chess.Board(fen)                                      │
│     history = last 2 (from_sq, to_sq) from move_history           │
│     current_planes = board_to_planes(board, history)              │
│                                                                    │
│  B. Candidate planes (M, 47, 8, 8)                               │
│     For each candidate move:                                       │
│       board_after = chess.Board(fen_after)                        │
│       cand_history = [(from_sq, to_sq)] + history[:1]             │
│       planes = board_to_planes(board_after, cand_history)         │
│     Pad to max_possible (40) with zeros                           │
│                                                                    │
│  C. Candidate scalars (M, 11) — ALL STM-NORMALIZED               │
│     For each candidate:                                            │
│       eval_stm = eval if White else -eval                         │
│       stm_win = white_win% if White else black_win%               │
│       stm_loss = black_win% if White else white_win%              │
│       [eval_stm/1000, stm_win, draw%, stm_loss,                  │
│        log1p(nodes)/20, depth/40, move_quality,                   │
│        piece_type, is_capture, is_check, is_checkmate]            │
│                                                                    │
│     move_quality = (eval_stm - worst_stm) / (best_stm - worst_stm)│
│                                                                    │
│  D. Tabular features (14) — STM-NORMALIZED                       │
│     [time_remaining/3600, w_elo/3000, b_elo/3000,                 │
│      elo_diff/1000, move_no/200, color(0|1),                      │
│      eval_before_stm/1000, stm_win_before, draw_before,           │
│      stm_loss_before, initial_time/3600, increment/60,            │
│      prev_move_was_capture, in_check]                              │
│                                                                    │
│  E. Possible mask (M,) — 1.0 for real moves, 0.0 for padding     │
│                                                                    │
│  Convert all to torch tensors, add batch dim, move to GPU         │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                  STEP 3: Run Model                                │
│                                                                    │
│  model = ChessMIMOModel()                                     │
│  model.load_state_dict(checkpoint['model_state_dict'])            │
│  model.eval()                                                      │
│                                                                    │
│  with torch.no_grad():                                            │
│      outputs = model(                                              │
│          current_planes,    # (1, 47, 8, 8)                       │
│          possible_planes,   # (1, M, 47, 8, 8)                   │
│          possible_scalars,  # (1, M, 11)                          │
│          possible_mask,     # (1, M)                              │
│          tabular,           # (1, 14)                             │
│          actual_idx=None,   # None at inference — we're predicting│
│      )                                                             │
│                                                                    │
│  NOTE: actual_idx=None means win_prob_before and win_prob_after   │
│  are NOT computed (they require knowing which move was played).   │
│  You get: move_logits, mistake_prob, time_spent.                  │
│                                                                    │
│  For win_prob_after: if you want to evaluate a SPECIFIC move,     │
│  pass its index as actual_idx and you get all 5 heads.            │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                  STEP 4: Interpret Results                         │
│                                                                    │
│  # Which move will the human play?                                │
│  move_probs = softmax(outputs['move_logits'][0])                  │
│  top_moves = move_probs.topk(5)                                   │
│  for prob, idx in zip(top_moves.values, top_moves.indices):       │
│      print(f"{candidates[idx].uci()} — {prob:.1%}")              │
│  # e.g.: d2d4 — 31.2%, b1c3 — 18.7%, d2d3 — 12.4%              │
│                                                                    │
│  # Is the predicted move likely a mistake?                        │
│  mistake_p = outputs['mistake_prob'][0].item()                    │
│  print(f"Mistake probability: {mistake_p:.1%}")                   │
│                                                                    │
│  # How long will they think?                                      │
│  time_log = outputs['time_spent'][0].item()                       │
│  time_secs = math.expm1(time_log)  # convert from log scale      │
│  print(f"Expected think time: {time_secs:.1f}s")                  │
│                                                                    │
│  # For WDL: run again with actual_idx pointing to a specific move │
│  # This answers "if they play Nf3, what's the win probability?"   │
│  outputs_nf3 = model(..., actual_idx=torch.tensor([nf3_idx]))     │
│  wdl_after = outputs_nf3['win_prob_after'][0]                     │
│  print(f"Win/Draw/Loss after Nf3: {wdl_after}")                  │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Minimal Inference Code

```python
#!/usr/bin/env python3
"""Predict the next move a human will play in a live chess game."""

import math
import chess
import torch
import numpy as np
from chess_mimo_model import ChessMIMOModel
from plane_codec import board_to_planes  # or use the inlined version in mimo_dataset


def parse_time_control(tc: str):
    """Parse '300+3' → (300.0, 3.0)"""
    if not tc or tc == '-':
        return 0.0, 0.0
    if '+' in tc:
        parts = tc.split('+')
        return float(parts[0]), float(parts[1])
    try:
        return float(tc), 0.0
    except ValueError:
        return 0.0, 0.0


def evaluate_candidates(board: chess.Board, engine) -> list:
    """
    Run engine on each legal move. Returns list of dicts:
    {move, from_sq, to_sq, piece, fen_after, eval, wdl, nodes, depth}
    
    You implement this with your engine wrapper (LC0 or Stockfish).
    """
    # Placeholder — replace with your engine integration
    candidates = []
    for move in board.legal_moves:
        board.push(move)
        # analysis = engine.analyse(board, chess.engine.Limit(nodes=1))
        # For now, dummy values:
        candidates.append({
            'move': move.uci(),
            'from_square': chess.square_name(move.from_square),
            'to_square': chess.square_name(move.to_square),
            'piece': board.piece_at(move.to_square).symbol().upper() if board.piece_at(move.to_square) else 'P',
            'fen_after': board.fen(),
            'eval': 0.0,
            'white_win_perc': 0.33,
            'draw_perc': 0.34,
            'black_win_perc': 0.33,
            'nodes': 1,
            'depth': 1,
        })
        board.pop()
    return candidates


def predict_move(
    fen: str,
    move_history: list,           # ["e2e4", "e7e5", ...]
    white_elo: int,
    black_elo: int,
    time_remaining: float,        # seconds for side to move
    time_control: str,            # e.g. "300+3"
    checkpoint_path: str,
    engine=None,                  # your chess engine
    max_possible: int = 40,
    device: str = 'cuda',
):
    """
    Predict what move a human will play, how long they'll think,
    and whether it's likely a mistake.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # --- Load model ---
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    model = ChessMIMOModel(
        cnn_channels=cfg.get('cnn_channels', 128),
        num_res_blocks=cfg.get('res_blocks', 6),
        tabular_dim=14,
        max_possible=max_possible,
        hidden_dim=cfg.get('hidden_dim', 256),
        move_scalar_dim=11,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # --- Set up board ---
    board = chess.Board(fen)
    color = 'White' if board.turn == chess.WHITE else 'Black'
    move_no = len(move_history) + 1

    # --- Parse history for planes ---
    history = []
    for uci in move_history[-2:]:
        try:
            m = chess.Move.from_uci(uci)
            history.append((m.from_square, m.to_square))
        except:
            pass

    # --- Current position planes ---
    current_planes = board_to_planes(board, history)

    # --- Evaluate candidates ---
    candidates = evaluate_candidates(board, engine)
    candidates = candidates[:max_possible]
    num_cands = len(candidates)

    # --- Evaluate current position (for tabular WDL_before) ---
    # In production, get this from engine too:
    eval_before = 0.0       # placeholder
    wdl_before = (0.33, 0.34, 0.33)  # placeholder

    # --- STM normalisation ---
    is_white = (color == 'White')

    eval_before_stm = eval_before if is_white else -eval_before
    stm_win_before = wdl_before[0] if is_white else wdl_before[2]
    stm_draw_before = wdl_before[1]
    stm_loss_before = wdl_before[2] if is_white else wdl_before[0]

    # --- Candidate planes + scalars ---
    poss_planes_list = []
    poss_scalars_list = []
    evals_stm = []

    for c in candidates:
        # Planes
        try:
            ba = chess.Board(c['fen_after'])
            from_sq = chess.parse_square(c['from_square'])
            to_sq = chess.parse_square(c['to_square'])
            c_hist = [(from_sq, to_sq)] + history[:1]
            poss_planes_list.append(board_to_planes(ba, c_hist))
        except:
            poss_planes_list.append(np.zeros((47, 8, 8), dtype=np.float32))

        # STM-normalised eval and WDL
        e_raw = c.get('eval', 0.0)
        e_stm = e_raw if is_white else -e_raw
        evals_stm.append(e_stm)

        if is_white:
            sw = c.get('white_win_perc', 0.33)
            sl = c.get('black_win_perc', 0.33)
        else:
            sw = c.get('black_win_perc', 0.33)
            sl = c.get('white_win_perc', 0.33)
        sd = c.get('draw_perc', 0.34)

        # is_capture
        to_sq_int = chess.parse_square(c['to_square'])
        is_cap = 1.0 if board.piece_at(to_sq_int) is not None else 0.0
        if board.ep_square == to_sq_int and c.get('piece', '').upper() == 'P':
            is_cap = 1.0

        # is_check / is_checkmate
        try:
            ba = chess.Board(c['fen_after'])
            is_chk = 1.0 if ba.is_check() else 0.0
            is_mate = 1.0 if ba.is_checkmate() else 0.0
        except:
            is_chk = is_mate = 0.0

        # piece_type
        piece_map = {'P': 1/6, 'N': 2/6, 'B': 3/6, 'R': 4/6, 'Q': 5/6, 'K': 1.0}
        pt = piece_map.get(c.get('piece', 'P').upper(), 1/6)

        nodes_raw = c.get('nodes', 1)

        poss_scalars_list.append(np.array([
            e_stm / 1000.0,
            sw, sd, sl,
            math.log1p(nodes_raw) / 20.0,
            c.get('depth', 1) / 40.0,
            0.0,  # move_quality placeholder — computed below
            pt, is_cap, is_chk, is_mate,
        ], dtype=np.float32))

    # Compute move_quality
    if evals_stm:
        best_e = max(evals_stm)
        worst_e = min(evals_stm)
        e_range = best_e - worst_e
        for i, e in enumerate(evals_stm):
            poss_scalars_list[i][6] = (e - worst_e) / e_range if e_range > 0 else 1.0

    # Pad to max_possible
    while len(poss_planes_list) < max_possible:
        poss_planes_list.append(np.zeros((47, 8, 8), dtype=np.float32))
        poss_scalars_list.append(np.zeros(11, dtype=np.float32))

    possible_mask = np.zeros(max_possible, dtype=np.float32)
    possible_mask[:num_cands] = 1.0

    # --- Tabular ---
    initial_time, increment = parse_time_control(time_control)

    # Previous move was capture
    prev_capture = 0.0
    if move_history:
        try:
            replay = chess.Board()
            for uci in move_history:
                m = chess.Move.from_uci(uci)
                if uci == move_history[-1]:
                    prev_capture = 1.0 if replay.is_capture(m) else 0.0
                replay.push(m)
        except:
            pass

    in_check = 1.0 if board.is_check() else 0.0

    tabular = np.array([
        time_remaining / 3600.0,
        white_elo / 3000.0,
        black_elo / 3000.0,
        (white_elo - black_elo) / 1000.0,
        move_no / 200.0,
        1.0 if is_white else 0.0,
        eval_before_stm / 1000.0,
        stm_win_before,
        stm_draw_before,
        stm_loss_before,
        initial_time / 3600.0,
        increment / 60.0,
        prev_capture,
        in_check,
    ], dtype=np.float32)

    # --- To tensors ---
    cp = torch.from_numpy(current_planes).unsqueeze(0).to(device)
    pp = torch.from_numpy(np.stack(poss_planes_list)).unsqueeze(0).to(device)
    ps = torch.from_numpy(np.stack(poss_scalars_list)).unsqueeze(0).to(device)
    pm = torch.from_numpy(possible_mask).unsqueeze(0).to(device)
    tab = torch.from_numpy(tabular).unsqueeze(0).to(device)

    # --- Run model ---
    with torch.no_grad():
        outputs = model(cp, pp, ps, pm, tab, actual_idx=None)

    # --- Parse outputs ---
    move_probs = torch.softmax(outputs['move_logits'][0], dim=0).cpu().numpy()
    mistake_p = outputs['mistake_prob'][0].item()
    time_log = outputs['time_spent'][0].item()
    time_secs = math.expm1(time_log)

    # Top-5 predictions
    top_indices = np.argsort(move_probs)[::-1][:5]
    predictions = []
    for idx in top_indices:
        if idx < num_cands:
            predictions.append({
                'move': candidates[idx]['move'],
                'probability': float(move_probs[idx]),
                'piece': candidates[idx].get('piece', '?'),
            })

    return {
        'top_moves': predictions,
        'mistake_probability': mistake_p,
        'expected_think_time_seconds': time_secs,
    }


# --- Example usage ---
if __name__ == '__main__':
    result = predict_move(
        fen="rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        move_history=["e2e4", "g8f6"],
        white_elo=1850,
        black_elo=1920,
        time_remaining=285.0,
        time_control="300+3",
        checkpoint_path="checkpoints/best.pt",
        device='cuda',
    )

    print("\\nPredicted moves:")
    for m in result['top_moves']:
        print(f"  {m['move']:6s}  {m['probability']:5.1%}  ({m['piece']})")
    print(f"\\nMistake probability: {result['mistake_probability']:.1%}")
    print(f"Expected think time: {result['expected_think_time_seconds']:.1f}s")
```

---

## Inference Modes

### Mode 1: "What will they play?" (actual_idx=None)

Pass `actual_idx=None`. You get:
- **move_logits** → probability distribution over candidates
- **mistake_prob** → P(the top-predicted move is a mistake)
- **time_spent** → expected thinking time

You do NOT get win_prob_before/after (those require knowing which move was played).

### Mode 2: "What if they play X?" (actual_idx=specific_move)

Pass `actual_idx=tensor([idx_of_move_X])`. You get all 5 heads:
- **move_logits** → same ranking
- **mistake_prob** → same (based on position, not the chosen move)
- **win_prob_before** → WDL before the move (actual move masked from attention)
- **win_prob_after** → WDL after move X (uses move X's embedding)
- **time_spent** → same

Use this to answer: "If they play Nf3, what's the game outlook?"

### Mode 3: "Evaluate all moves" (loop)

Run Mode 2 for each candidate to get a full analysis:

```python
for i, cand in enumerate(candidates):
    outputs = model(..., actual_idx=torch.tensor([i]))
    wdl = outputs['win_prob_after'][0].cpu().numpy()
    print(f"{cand['move']}: win={wdl[0]:.1%} draw={wdl[1]:.1%} loss={wdl[2]:.1%}")
```

This is M forward passes but each is fast (~5ms on GPU).

---

## Latency Budget (RTX 4090)

| Step | Time | Notes |
|------|------|-------|
| Engine analysis (LC0 batch) | ~50ms | All candidates in one batch, nodes=1 |
| Engine analysis (SF depth=14) | ~2-5s | Sequential per candidate |
| Plane generation (40 candidates) | ~20ms | CPU, python-chess |
| Model forward pass | ~5ms | GPU, single batch |
| **Total (LC0)** | **~75ms** | Fast enough for real-time |
| **Total (SF)** | **~2-5s** | Acceptable for analysis |

### Optimisation tips:
- **Batch engine calls** — LC0 supports batch analysis, much faster than per-move
- **Cache planes** — if analyzing multiple positions in the same game, reuse prior planes
- **ONNX export** — `torch.onnx.export()` for production deployment without PyTorch overhead
- **TensorRT** — 2-3× speedup on NVIDIA GPUs for the CNN forward pass
- **Pre-compute candidates** — start engine analysis before the opponent moves (on their likely responses)

---

## Adapting for Different Engines

The model was trained on LC0 nodes=1 evaluations. If you use a different engine at inference:

| Engine | Compatibility | Notes |
|--------|--------------|-------|
| LC0 nodes=1 | ✓ Perfect | Same as training data |
| LC0 nodes>1 | ⚠ Good | Slightly different eval distribution; model should still work |
| Stockfish | ⚠ Fair | Different eval scale; consider training a separate model on SF data |
| Mixed (LC0 + SF) | ⚠ Fair | Eval features may confuse model if it learned LC0-specific patterns |

When the Stockfish depth=14 pass completes, you could:
1. Train a separate model on SF data
2. Add `evaluated_by` as a categorical feature so one model handles both
3. Use SF data for the second pass (AB test: LC0-only vs LC0+SF)
