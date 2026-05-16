#!/usr/bin/env python3
"""
mimo_dataset_opus.py — Dataset builder for the MIMO Opus chess model.

Loads from Parquet (games + actual_moves + possible_moves), generates
47-plane CNN encodings for the current position AND every candidate move's
resulting position, and outputs a compressed .npz ready for training.

Key improvements over v3:
    - Per-move scalar features (eval, WDL, nodes, depth) alongside planes
    - Multiprocessing for plane generation (CPU-bound with python-chess)
    - Train / val / test split by game_id (no same-game leakage)
    - Improved mistake detection: compares played-move eval to BEST candidate
    - 10-dim tabular vector — no time_spent, no dead padding zeros
    - Proper history parsing from game_to_position for temporal planes

Usage:
    python mimo_dataset_opus.py \
        --moves  data/actual_moves.parquet \
        --games  data/games.parquet \
        --possible data/possible_moves.parquet \
        --output-dir data/mimo_opus_dataset \
        --max-possible 40 \
        --min-elo 0 \
        --workers 8
"""

import os
import sys
import argparse
import hashlib
import json
import math
import traceback
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Plane codec (inlined from plane_codec.py for self-containment)
# ---------------------------------------------------------------------------

def board_to_planes(board: chess.Board,
                    history: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    """
    Convert a chess.Board to 47×8×8 plane representation.

    Planes 0-11:  current position pieces (P N B R Q K white, then black)
    Planes 12-23: t-1 position pieces
    Planes 24-35: t-2 position pieces
    Planes 36-37: last move from/to (t-1)
    Planes 38-39: last move from/to (t-2)
    Plane  40:    side to move (1.0 = white)
    Planes 41-44: castling rights (WK, WQ, BK, BQ)
    Plane  45:    en passant target square
    Plane  46:    fifty-move counter / 100
    """
    planes = np.zeros((47, 8, 8), dtype=np.float32)

    # --- Piece planes for current position (0-11) ---
    for pt in range(1, 7):
        for color in (chess.WHITE, chess.BLACK):
            idx = (pt - 1) + 6 * color
            for sq in board.pieces(pt, color):
                planes[idx, 7 - sq // 8, sq % 8] = 1.0

    # --- History positions & move-from/to planes ---
    if history:
        try:
            temp = board.copy()
            if temp.move_stack and len(history) >= 1:
                temp.pop()
                _fill_piece_planes(temp, planes, offset=12)
        except Exception:
            pass
        try:
            temp2 = board.copy()
            if len(temp2.move_stack) >= 2 and len(history) >= 2:
                temp2.pop()
                temp2.pop()
                _fill_piece_planes(temp2, planes, offset=24)
        except Exception:
            pass

        if len(history) >= 1:
            fr, to = history[0]
            if fr is not None:
                planes[36, 7 - fr // 8, fr % 8] = 1.0
                planes[37, 7 - to // 8, to % 8] = 1.0
        if len(history) >= 2:
            fr, to = history[1]
            if fr is not None:
                planes[38, 7 - fr // 8, fr % 8] = 1.0
                planes[39, 7 - to // 8, to % 8] = 1.0

    # --- Global planes ---
    planes[40, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[41, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[42, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[43, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[44, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    if board.ep_square is not None:
        planes[45, 7 - board.ep_square // 8, board.ep_square % 8] = 1.0
    planes[46, :, :] = min(board.halfmove_clock, 100) / 100.0

    return planes


def _fill_piece_planes(board: chess.Board, planes: np.ndarray, offset: int):
    for pt in range(1, 7):
        for color in (chess.WHITE, chess.BLACK):
            idx = offset + (pt - 1) + 6 * color
            for sq in board.pieces(pt, color):
                planes[idx, 7 - sq // 8, sq % 8] = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_game_to_position(gtp_str: str) -> List[Tuple[int, int]]:
    """Parse 'e2e4,d7d5,...' string into list of (from_sq, to_sq) tuples."""
    if not gtp_str or (isinstance(gtp_str, float) and math.isnan(gtp_str)):
        return []
    moves = []
    for tok in str(gtp_str).split(','):
        tok = tok.strip()
        if len(tok) >= 4:
            try:
                moves.append((chess.parse_square(tok[:2]), chess.parse_square(tok[2:4])))
            except ValueError:
                pass
    return moves


def result_to_wdl(result: str, color: str) -> np.ndarray:
    """Convert game result to [win, draw, loss] from color's perspective."""
    if result == '1-0':
        return np.array([1., 0., 0.], dtype=np.float32) if color == 'White' \
            else np.array([0., 0., 1.], dtype=np.float32)
    elif result == '0-1':
        return np.array([1., 0., 0.], dtype=np.float32) if color == 'Black' \
            else np.array([0., 0., 1.], dtype=np.float32)
    else:
        return np.array([0., 1., 0.], dtype=np.float32)


def _safe_float(val, default: float = 0.0) -> float:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def parse_time_control(tc) -> Tuple[float, float]:
    """Parse '300+3' → (initial_seconds, increment_seconds)."""
    if not tc or (isinstance(tc, float) and math.isnan(tc)):
        return 0.0, 0.0
    tc = str(tc).strip()
    if tc == '-':
        return 0.0, 0.0
    if '+' in tc:
        parts = tc.split('+')
        try:
            return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            return 0.0, 0.0
    if '/' in tc:
        parts = tc.split('/')
        try:
            return float(parts[1]) if len(parts) > 1 else 0.0, 0.0
        except (ValueError, IndexError):
            return 0.0, 0.0
    try:
        return float(tc), 0.0
    except ValueError:
        return 0.0, 0.0


def detect_prev_capture(game_to_position_str: str) -> float:
    """Replay game_to_position moves; return 1.0 if the last move was a capture."""
    moves = parse_game_to_position(game_to_position_str)
    if not moves:
        return 0.0
    try:
        replay = chess.Board()
        was_capture = False
        for from_sq, to_sq in moves:
            matched = False
            for legal in replay.legal_moves:
                if legal.from_square == from_sq and legal.to_square == to_sq:
                    was_capture = replay.is_capture(legal)
                    replay.push(legal)
                    matched = True
                    break
            if not matched:
                # Try any promotion variant
                for legal in replay.legal_moves:
                    if legal.from_square == from_sq and legal.to_square == to_sq:
                        was_capture = replay.is_capture(legal)
                        replay.push(legal)
                        break
        return 1.0 if was_capture else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Single-example builder (used by multiprocessing pool)
# ---------------------------------------------------------------------------

def build_one_example(args: Tuple) -> Optional[Dict[str, np.ndarray]]:
    """
    Build one training example from a move row + game row + possible moves.

    Returns None on failure (bad FEN, missing data, etc.).
    """
    move_dict, game_dict, possibles, max_possible = args

    try:
        fen_before = move_dict['fen_before']
        board = chess.Board(fen_before)
    except Exception:
        return None

    if not possibles:
        return None

    color = move_dict.get('color', 'White')
    game_result = game_dict.get('result', '1/2-1/2')
    w_elo = _safe_float(game_dict.get('white_elo'), 1500)
    b_elo = _safe_float(game_dict.get('black_elo'), 1500)

    # --- Current position planes ---
    history_tuples = parse_game_to_position(move_dict.get('game_to_position', ''))
    recent_history = history_tuples[-2:] if history_tuples else []
    try:
        current_planes = board_to_planes(board, recent_history)
    except Exception:
        return None

    # --- Sort possible moves by eval (best first) and cap ---
    possibles = sorted(possibles, key=lambda x: _safe_float(x.get('eval'), -99999), reverse=True)
    possibles = possibles[:max_possible]
    num_possible = len(possibles)

    # --- Compute STM-normalized evals for all candidates (needed for move_quality) ---
    evals_stm = []
    for pm in possibles:
        e = _safe_float(pm.get('eval'), 0)
        evals_stm.append(e if color == 'White' else -e)
    best_eval_stm = max(evals_stm) if evals_stm else 0.0
    worst_eval_stm = min(evals_stm) if evals_stm else 0.0
    eval_range = best_eval_stm - worst_eval_stm

    # --- Encode each possible move ---
    poss_planes_list: List[np.ndarray] = []
    poss_scalars_list: List[np.ndarray] = []
    piece_map = {'P': 1/6, 'N': 2/6, 'B': 3/6, 'R': 4/6, 'Q': 5/6, 'K': 1.0}

    for i, pm in enumerate(possibles):
        # Plane encoding for position after this candidate move
        try:
            board_after = chess.Board(pm['fen_after'])
            from_sq = chess.parse_square(pm['from_square'])
            to_sq = chess.parse_square(pm['to_square'])
            poss_hist = [(from_sq, to_sq)] + recent_history[:1]
            poss_planes_list.append(board_to_planes(board_after, poss_hist))
        except Exception:
            board_after = None
            poss_planes_list.append(np.zeros((47, 8, 8), dtype=np.float32))

        # Scalar features for this candidate (STM perspective)
        pm_eval_stm = evals_stm[i]

        if color == 'White':
            pm_stm_win  = _safe_float(pm.get('white_win_perc'), 0.33)
            pm_stm_draw = _safe_float(pm.get('draw_perc'), 0.34)
            pm_stm_loss = _safe_float(pm.get('black_win_perc'), 0.33)
        else:
            pm_stm_win  = _safe_float(pm.get('black_win_perc'), 0.33)
            pm_stm_draw = _safe_float(pm.get('draw_perc'), 0.34)
            pm_stm_loss = _safe_float(pm.get('white_win_perc'), 0.33)

        nodes_raw = _safe_float(pm.get('nodes'), 1)

        # move_quality: % of way from worst to best eval (STM)
        if eval_range > 0:
            move_quality = (pm_eval_stm - worst_eval_stm) / eval_range
        else:
            move_quality = 1.0  # all moves equal

        # piece_type: P=1/6, N=2/6, B=3/6, R=4/6, Q=5/6, K=1.0
        piece_val = piece_map.get(str(pm.get('piece', 'P')).upper(), 1/6)

        # is_capture: piece on target square or en passant
        try:
            to_sq_int = chess.parse_square(pm['to_square'])
            is_capture = 1.0 if board.piece_at(to_sq_int) is not None else 0.0
            if board.ep_square == to_sq_int and str(pm.get('piece', '')).upper() == 'P':
                is_capture = 1.0
        except Exception:
            is_capture = 0.0

        # is_check, is_checkmate: from board_after
        is_check = 0.0
        is_checkmate = 0.0
        if board_after is not None:
            try:
                is_check = 1.0 if board_after.is_check() else 0.0
                is_checkmate = 1.0 if board_after.is_checkmate() else 0.0
            except Exception:
                pass

        poss_scalars_list.append(np.array([
            pm_eval_stm / 1000.0,           # 0: eval (STM)
            pm_stm_win,                       # 1: stm_win_perc
            pm_stm_draw,                      # 2: draw_perc
            pm_stm_loss,                      # 3: stm_loss_perc
            math.log1p(nodes_raw) / 20.0,    # 4: log-scaled nodes
            _safe_float(pm.get('depth'), 20) / 40.0,  # 5: depth
            move_quality,                     # 6: move_quality (0-1)
            piece_val,                        # 7: piece_type
            is_capture,                       # 8: is_capture
            is_check,                         # 9: is_check
            is_checkmate,                     # 10: is_checkmate
        ], dtype=np.float32))

    # --- Pad to max_possible ---
    while len(poss_planes_list) < max_possible:
        poss_planes_list.append(np.zeros((47, 8, 8), dtype=np.float32))
        poss_scalars_list.append(np.zeros(11, dtype=np.float32))

    possible_mask = np.zeros(max_possible, dtype=np.float32)
    possible_mask[:num_possible] = 1.0

    # --- Tabular features (14-dim, NO time_spent, NO padding) ---
    # WDL and eval normalised to side-to-move (STM) perspective:
    #   positive eval = good for STM, stm_win = STM's winning chance
    eval_raw = _safe_float(move_dict.get('eval_before'), 0)
    eval_stm = eval_raw if color == 'White' else -eval_raw

    if color == 'White':
        stm_win_before  = _safe_float(move_dict.get('white_win_perc_before'), 0.33)
        stm_draw_before = _safe_float(move_dict.get('draw_perc_before'), 0.34)
        stm_loss_before = _safe_float(move_dict.get('black_win_perc_before'), 0.33)
    else:
        stm_win_before  = _safe_float(move_dict.get('black_win_perc_before'), 0.33)
        stm_draw_before = _safe_float(move_dict.get('draw_perc_before'), 0.34)
        stm_loss_before = _safe_float(move_dict.get('white_win_perc_before'), 0.33)

    # Parse time control from games table
    initial_time, increment = parse_time_control(game_dict.get('time_control', ''))

    # Previous move was capture (replay game_to_position)
    prev_capture = detect_prev_capture(move_dict.get('game_to_position', ''))

    # Currently in check
    in_check = 1.0 if board.is_check() else 0.0

    # --- Position complexity metrics (computed from already-available data) ---
    # eval_std: standard deviation of STM evals across candidates
    eval_std = float(np.std(evals_stm)) / 1000.0 if len(evals_stm) > 1 else 0.0
    # num_captures: fraction of candidates that are captures
    captures_arr = [s[8] for s in poss_scalars_list[:num_possible]]
    num_captures = sum(captures_arr) / max(num_possible, 1)
    # num_checks: fraction of candidates that give check
    checks_arr = [s[9] for s in poss_scalars_list[:num_possible]]
    num_checks = sum(checks_arr) / max(num_possible, 1)
    # num_candidates: normalized count of legal moves
    num_candidates = num_possible / max_possible

    tabular = np.array([
        _safe_float(move_dict.get('time_remaining')) / 3600.0,  # 0
        w_elo / 3000.0,                                          # 1
        b_elo / 3000.0,                                          # 2
        (w_elo - b_elo) / 1000.0,                                # 3
        _safe_float(move_dict.get('move_no')) / 200.0,           # 4
        1.0 if color == 'White' else 0.0,                        # 5
        eval_stm / 1000.0,                                       # 6
        stm_win_before,                                           # 7
        stm_draw_before,                                          # 8
        stm_loss_before,                                          # 9
        initial_time / 3600.0,                                    # 10
        increment / 60.0,                                         # 11
        prev_capture,                                             # 12
        in_check,                                                 # 13
        eval_std,                                                 # 14: position sharpness
        num_captures,                                             # 15: tactical complexity
        num_checks,                                               # 16: check pressure
        num_candidates,                                           # 17: position openness
    ], dtype=np.float32)

    # --- Find actual move index among candidates ---
    actual_uci = move_dict.get('move', '')
    actual_idx = -1
    for i, pm in enumerate(possibles):
        if pm.get('move') == actual_uci:
            actual_idx = i
            break

    # --- Mistake detection (WDL-based) ---
    # Uses W + 0.5*D (expected score) drop between best and played move.
    # Also flags outcome-shifting moves (W→D, W→L, D→L) if drop > 5%.
    is_mistake = 0.0
    if actual_idx >= 0 and len(possibles) > 0:
        # Find best move by expected score (W + 0.5*D), STM perspective
        def _expected_score(pm_dict, clr):
            if clr == 'White':
                w = _safe_float(pm_dict.get('white_win_perc'), 0.33)
                d = _safe_float(pm_dict.get('draw_perc'), 0.34)
            else:
                w = _safe_float(pm_dict.get('black_win_perc'), 0.33)
                d = _safe_float(pm_dict.get('draw_perc'), 0.34)
            return w + 0.5 * d

        def _outcome_class(pm_dict, clr):
            """Return dominant outcome: 'W', 'D', or 'L' for STM."""
            if clr == 'White':
                w = _safe_float(pm_dict.get('white_win_perc'), 0.33)
                d = _safe_float(pm_dict.get('draw_perc'), 0.34)
                l = _safe_float(pm_dict.get('black_win_perc'), 0.33)
            else:
                w = _safe_float(pm_dict.get('black_win_perc'), 0.33)
                d = _safe_float(pm_dict.get('draw_perc'), 0.34)
                l = _safe_float(pm_dict.get('white_win_perc'), 0.33)
            mx = max(w, d, l)
            if mx == w:
                return 'W'
            elif mx == d:
                return 'D'
            return 'L'

        best_idx = max(range(len(possibles)), key=lambda i: _expected_score(possibles[i], color))
        best_es = _expected_score(possibles[best_idx], color)
        played_es = _expected_score(possibles[actual_idx], color)
        drop = best_es - played_es

        # Elo-adaptive threshold on expected score drop
        avg_elo = (w_elo + b_elo) / 2
        threshold = 0.20 if avg_elo < 1500 else (0.15 if avg_elo < 2500 else 0.10)

        if drop > threshold:
            is_mistake = 1.0

        # Outcome shift: if most likely result worsens (W→D, W→L, D→L)
        # and drop > 5%, always flag as mistake
        if drop > 0.05 and is_mistake == 0.0:
            best_outcome = _outcome_class(possibles[best_idx], color)
            played_outcome = _outcome_class(possibles[actual_idx], color)
            outcome_rank = {'W': 2, 'D': 1, 'L': 0}
            if outcome_rank[played_outcome] < outcome_rank[best_outcome]:
                is_mistake = 1.0

    # --- WDL targets ---
    wdl_before = result_to_wdl(game_result, color)
    color_after = 'Black' if color == 'White' else 'White'
    wdl_after = result_to_wdl(game_result, color_after)

    # --- Time spent target (log scale) ---
    raw_ts = max(0.0, _safe_float(move_dict.get('time_spent')))
    time_spent_log = np.float32(math.log1p(raw_ts))

    return {
        'current_planes': current_planes,                                 # (47, 8, 8)
        'possible_planes': np.stack(poss_planes_list),                    # (M, 47, 8, 8)
        'possible_scalars': np.stack(poss_scalars_list),                  # (M, 6)
        'possible_mask': possible_mask,                                   # (M,)
        'tabular': tabular,                                               # (10,)
        'actual_idx': np.int64(actual_idx),
        'is_mistake': np.float32(is_mistake),
        'win_prob_before': wdl_before,                                    # (3,)
        'win_prob_after': wdl_after,                                      # (3,)
        'time_spent_log': time_spent_log,
    }


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    data_dir: str,
    output_dir: str,
    max_possible: int = 40,
    min_elo: int = 0,
    max_elo: int = 0,
    val_frac: float = 0.1,
    test_frac: float = 0.05,
    workers: int = 4,
    seed: int = 42,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load Parquet tables ----
    print("[1/5] Loading parquet tables …")

    def _load_table(root: Path, table_name: str) -> pd.DataFrame:
        """Find and merge all parquet files for a table across worker dirs."""
        parquet_files = sorted(root.rglob(f'{table_name}/*.parquet'))
        if not parquet_files:
            # Try flat layout: root/table_name.parquet
            flat = root / f'{table_name}.parquet'
            if flat.is_file():
                return pq.read_table(str(flat)).to_pandas()
            raise FileNotFoundError(
                f"No parquet files found for '{table_name}' under {root}")
        tables = [pq.read_table(str(f)) for f in parquet_files]
        combined = pa.concat_tables(tables)
        print(f"       {table_name}: merged {len(parquet_files)} files "
              f"from {len(set(f.parent.parent.name for f in parquet_files))} workers")
        return combined.to_pandas()

    root = Path(data_dir)
    moves_df = _load_table(root, 'moves')
    games_df = _load_table(root, 'games').set_index('game_id')
    poss_df  = _load_table(root, 'possible_moves')
    print(f"       moves={len(moves_df):,}  games={len(games_df):,}  possible={len(poss_df):,}")

    # ---- Elo filter ----
    if min_elo > 0 or max_elo > 0:
        elo_mask = pd.Series(True, index=games_df.index)
        if min_elo > 0:
            elo_mask &= (games_df['white_elo'] >= min_elo) & (games_df['black_elo'] >= min_elo)
        if max_elo > 0:
            elo_mask &= (games_df['white_elo'] <= max_elo) & (games_df['black_elo'] <= max_elo)
        valid_games = games_df[elo_mask].index
        moves_df = moves_df[moves_df['game_id'].isin(valid_games)]
        poss_df  = poss_df[poss_df['game_id'].isin(valid_games)]
        filter_desc = []
        if min_elo > 0:
            filter_desc.append(f"Elo≥{min_elo}")
        if max_elo > 0:
            filter_desc.append(f"Elo≤{max_elo}")
        print(f"       After {' & '.join(filter_desc)} filter: {len(moves_df):,} moves")

    # ---- Index possible moves by (game_id, move_no) ----
    print("[2/5] Indexing possible moves …")
    poss_index: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for _, row in poss_df.iterrows():
        poss_index[(row['game_id'], row['move_no'])].append(row.to_dict())
    print(f"       {len(poss_index):,} (game, move) keys")

    # ---- Split by game_id (not by row!) ----
    print("[3/5] Splitting by game_id …")
    rng = np.random.RandomState(seed)
    unique_games = moves_df['game_id'].unique()
    rng.shuffle(unique_games)
    n_val  = max(1, int(len(unique_games) * val_frac))
    n_test = max(1, int(len(unique_games) * test_frac))
    test_games = set(unique_games[:n_test])
    val_games  = set(unique_games[n_test:n_test + n_val])
    train_games = set(unique_games[n_test + n_val:])
    print(f"       train={len(train_games)}  val={len(val_games)}  test={len(test_games)}")

    # ---- Build examples per split ----
    for split_name, split_gids in [('train', train_games), ('val', val_games), ('test', test_games)]:
        split_moves = moves_df[moves_df['game_id'].isin(split_gids)]
        print(f"\n[4/5] Building {split_name}: {len(split_moves):,} moves …")

        # Prepare args for pool workers
        tasks = []
        for _, mrow in split_moves.iterrows():
            gid = mrow['game_id']
            mno = mrow['move_no']
            if gid not in games_df.index:
                continue
            possibles = poss_index.get((gid, mno), [])
            if not possibles:
                continue
            tasks.append((mrow.to_dict(), games_df.loc[gid].to_dict(), possibles, max_possible))

        # Process (multiprocessing for plane generation)
        results: List[Optional[Dict]] = []
        if workers > 1:
            with Pool(workers) as pool:
                results = pool.map(build_one_example, tasks, chunksize=64)
        else:
            results = [build_one_example(t) for t in tasks]

        examples = [r for r in results if r is not None]
        print(f"       Built {len(examples):,} valid examples (dropped {len(tasks) - len(examples):,})")

        if not examples:
            print(f"       ⚠ Skipping {split_name} — no valid examples")
            continue

        # Stack and save
        npz_path = out / f'{split_name}.npz'
        np.savez_compressed(
            npz_path,
            current_planes  = np.stack([e['current_planes']  for e in examples]),
            possible_planes = np.stack([e['possible_planes'] for e in examples]),
            possible_scalars= np.stack([e['possible_scalars'] for e in examples]),
            possible_mask   = np.stack([e['possible_mask']   for e in examples]),
            tabular         = np.stack([e['tabular']         for e in examples]),
            actual_idx      = np.array([e['actual_idx']      for e in examples]),
            is_mistake      = np.array([e['is_mistake']      for e in examples]),
            win_prob_before = np.stack([e['win_prob_before'] for e in examples]),
            win_prob_after  = np.stack([e['win_prob_after']  for e in examples]),
            time_spent_log  = np.array([e['time_spent_log']  for e in examples]),
        )
        sz_mb = os.path.getsize(npz_path) / (1024 * 1024)
        print(f"       Saved {npz_path}  ({sz_mb:.1f} MB)")

    # ---- Save config ----
    config = {
        'max_possible': max_possible,
        'min_elo': min_elo,
        'max_elo': max_elo,
        'tabular_dim': 18,
        'move_scalar_dim': 11,
        'tabular_features': [
            'time_remaining/3600',
            'white_elo/3000',
            'black_elo/3000',
            'elo_diff/1000',
            'move_no/200',
            'color_01',
            'eval_stm_before/1000',
            'stm_win_before',
            'draw_perc_before',
            'stm_loss_before',
            'initial_time/3600',
            'increment/60',
            'prev_move_was_capture',
            'in_check',
            'eval_std/1000',
            'num_captures_frac',
            'num_checks_frac',
            'num_candidates_frac',
        ],
        'move_scalar_features': [
            'eval_stm/1000',
            'stm_win_perc',
            'draw_perc',
            'stm_loss_perc',
            'log1p(nodes)/20',
            'depth/40',
            'move_quality',
            'piece_type',
            'is_capture',
            'is_check',
            'is_checkmate',
        ],
        'splits': {
            'val_frac': val_frac,
            'test_frac': test_frac,
            'seed': seed,
        },
    }
    with open(out / 'dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n[5/5] Saved dataset_config.json")
    print("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build MIMO Opus chess dataset')
    parser.add_argument('--data-dir',   required=True,
                        help='Root output dir (e.g. output/lc0/791556). Finds worker_*/games/, worker_*/moves/, worker_*/possible_moves/ automatically.')
    parser.add_argument('--output-dir', required=True, help='Output directory for .npz splits')
    parser.add_argument('--max-possible', type=int, default=40,   help='Max candidate moves per position')
    parser.add_argument('--min-elo',      type=int, default=0,    help='Min Elo for both players (0=off)')
    parser.add_argument('--max-elo',      type=int, default=0,    help='Max Elo for both players (0=off)')
    parser.add_argument('--val-frac',     type=float, default=0.10)
    parser.add_argument('--test-frac',    type=float, default=0.05)
    parser.add_argument('--workers',      type=int, default=4,    help='Multiprocessing workers')
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    build_dataset(
        args.data_dir,
        args.output_dir,
        max_possible=args.max_possible,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        workers=args.workers,
        seed=args.seed,
    )
