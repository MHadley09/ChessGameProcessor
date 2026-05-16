#!/usr/bin/env python3
"""
mimo_dataset.py — Compact dataset for the MIMO chess model.

Reads directly from Parquet files produced by the LC0 processing pipeline
(games + moves + possible_moves). Stores only compact data per position:
FEN strings, candidate UCI moves, tabular scalars, per-candidate scalars,
and labels. Board planes (47×8×8) are constructed on-the-fly in __getitem__
from FEN using python-chess — never pre-computed or stored.

Storage: ~2 KB/position (vs ~495 KB with pre-stored planes).
At 45M positions: ~90 GB compact vs ~22 TB with planes.

Two modes:
  1. --build: Read parquet → produce train/val/test .npz with compact data
  2. MIMOCompactDataset: PyTorch Dataset that loads .npz and builds planes on-the-fly

Usage:
    python mimo_dataset.py \\
        --data-dir  output/lc0/791556 \\
        --output-dir data/mimo_dataset \\
        --max-possible 40 \\
        --workers 8
"""

import argparse
import json
import math
import os
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
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Plane codec (inlined for self-containment)
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

    for pt in range(1, 7):
        for color in (chess.WHITE, chess.BLACK):
            idx = (pt - 1) + 6 * color
            for sq in board.pieces(pt, color):
                planes[idx, 7 - sq // 8, sq % 8] = 1.0

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
                for legal in replay.legal_moves:
                    if legal.from_square == from_sq and legal.to_square == to_sq:
                        was_capture = replay.is_capture(legal)
                        replay.push(legal)
                        break
        return 1.0 if was_capture else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Single-example builder — compact version (NO planes stored)
# ---------------------------------------------------------------------------

def build_one_compact(args: Tuple) -> Optional[Dict[str, Any]]:
    """
    Build one compact training example.  Stores FEN strings and UCI moves
    instead of pre-computed planes.  Planes are built at training time.

    Returns None on failure.
    """
    move_dict, game_dict, possibles, max_possible = args

    fen_before = move_dict.get('fen_before', '')
    try:
        board = chess.Board(fen_before)
    except Exception:
        return None

    if not possibles:
        return None

    color = move_dict.get('color', 'White')
    game_result = game_dict.get('result', '1/2-1/2')
    w_elo = _safe_float(game_dict.get('white_elo'), 1500)
    b_elo = _safe_float(game_dict.get('black_elo'), 1500)

    # --- Sort possible moves by eval (best first) and cap ---
    possibles = sorted(possibles, key=lambda x: _safe_float(x.get('eval'), -99999), reverse=True)
    possibles = possibles[:max_possible]
    num_possible = len(possibles)

    # --- Compute STM-normalized evals ---
    evals_stm = []
    for pm in possibles:
        e = _safe_float(pm.get('eval'), 0)
        evals_stm.append(e if color == 'White' else -e)
    best_eval_stm = max(evals_stm) if evals_stm else 0.0
    worst_eval_stm = min(evals_stm) if evals_stm else 0.0
    eval_range = best_eval_stm - worst_eval_stm

    # --- Encode per-candidate scalars (11-dim) + store UCI + FEN ---
    poss_scalars_list: List[np.ndarray] = []
    poss_uci_list: List[str] = []
    poss_fen_after_list: List[str] = []
    piece_map = {'P': 1/6, 'N': 2/6, 'B': 3/6, 'R': 4/6, 'Q': 5/6, 'K': 1.0}

    for i, pm in enumerate(possibles):
        poss_uci_list.append(pm.get('move', ''))
        poss_fen_after_list.append(pm.get('fen_after', ''))

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
        move_quality = ((pm_eval_stm - worst_eval_stm) / eval_range) if eval_range > 0 else 1.0
        piece_val = piece_map.get(str(pm.get('piece', 'P')).upper(), 1/6)

        try:
            to_sq_int = chess.parse_square(pm['to_square'])
            is_capture = 1.0 if board.piece_at(to_sq_int) is not None else 0.0
            if board.ep_square == to_sq_int and str(pm.get('piece', '')).upper() == 'P':
                is_capture = 1.0
        except Exception:
            is_capture = 0.0

        is_check = 0.0
        is_checkmate = 0.0
        try:
            board_after = chess.Board(pm.get('fen_after', ''))
            is_check = 1.0 if board_after.is_check() else 0.0
            is_checkmate = 1.0 if board_after.is_checkmate() else 0.0
        except Exception:
            pass

        poss_scalars_list.append(np.array([
            pm_eval_stm / 1000.0,
            pm_stm_win,
            pm_stm_draw,
            pm_stm_loss,
            math.log1p(nodes_raw) / 20.0,
            _safe_float(pm.get('depth'), 20) / 40.0,
            move_quality,
            piece_val,
            is_capture,
            is_check,
            is_checkmate,
        ], dtype=np.float32))

    # Pad scalars to max_possible
    while len(poss_scalars_list) < max_possible:
        poss_scalars_list.append(np.zeros(11, dtype=np.float32))
        poss_uci_list.append('')
        poss_fen_after_list.append('')

    possible_mask = np.zeros(max_possible, dtype=np.float32)
    possible_mask[:num_possible] = 1.0

    # --- Tabular features (18-dim) ---
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

    initial_time, increment = parse_time_control(game_dict.get('time_control', ''))
    prev_capture = detect_prev_capture(move_dict.get('game_to_position', ''))
    in_check = 1.0 if board.is_check() else 0.0

    eval_std = float(np.std(evals_stm)) / 1000.0 if len(evals_stm) > 1 else 0.0
    captures_arr = [s[8] for s in poss_scalars_list[:num_possible]]
    num_captures = sum(captures_arr) / max(num_possible, 1)
    checks_arr = [s[9] for s in poss_scalars_list[:num_possible]]
    num_checks = sum(checks_arr) / max(num_possible, 1)
    num_candidates = num_possible / max_possible

    tabular = np.array([
        _safe_float(move_dict.get('time_remaining')) / 3600.0,
        w_elo / 3000.0,
        b_elo / 3000.0,
        (w_elo - b_elo) / 1000.0,
        _safe_float(move_dict.get('move_no')) / 200.0,
        1.0 if color == 'White' else 0.0,
        eval_stm / 1000.0,
        stm_win_before,
        stm_draw_before,
        stm_loss_before,
        initial_time / 3600.0,
        increment / 60.0,
        prev_capture,
        in_check,
        eval_std,
        num_captures,
        num_checks,
        num_candidates,
    ], dtype=np.float32)

    # --- Actual move index ---
    actual_uci = move_dict.get('move', '')
    actual_idx = -1
    for i, pm in enumerate(possibles):
        if pm.get('move') == actual_uci:
            actual_idx = i
            break

    # --- Mistake detection (WDL-based) ---
    is_mistake = 0.0
    if actual_idx >= 0 and len(possibles) > 0:
        def _expected_score(pm_dict, clr):
            if clr == 'White':
                w = _safe_float(pm_dict.get('white_win_perc'), 0.33)
                d = _safe_float(pm_dict.get('draw_perc'), 0.34)
            else:
                w = _safe_float(pm_dict.get('black_win_perc'), 0.33)
                d = _safe_float(pm_dict.get('draw_perc'), 0.34)
            return w + 0.5 * d

        def _outcome_class(pm_dict, clr):
            if clr == 'White':
                w = _safe_float(pm_dict.get('white_win_perc'), 0.33)
                d = _safe_float(pm_dict.get('draw_perc'), 0.34)
                l = _safe_float(pm_dict.get('black_win_perc'), 0.33)
            else:
                w = _safe_float(pm_dict.get('black_win_perc'), 0.33)
                d = _safe_float(pm_dict.get('draw_perc'), 0.34)
                l = _safe_float(pm_dict.get('white_win_perc'), 0.33)
            mx = max(w, d, l)
            if mx == w: return 'W'
            elif mx == d: return 'D'
            return 'L'

        best_idx = max(range(len(possibles)), key=lambda i: _expected_score(possibles[i], color))
        best_es = _expected_score(possibles[best_idx], color)
        played_es = _expected_score(possibles[actual_idx], color)
        drop = best_es - played_es

        avg_elo = (w_elo + b_elo) / 2
        threshold = 0.20 if avg_elo < 1500 else (0.15 if avg_elo < 2500 else 0.10)

        if drop > threshold:
            is_mistake = 1.0

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

    # --- game_to_position for history reconstruction at train time ---
    gtp = str(move_dict.get('game_to_position', '')) if move_dict.get('game_to_position') else ''

    return {
        # Compact: strings instead of planes
        'fen_before': fen_before,
        'game_to_position': gtp,
        'possible_uci': poss_uci_list,         # list of max_possible UCI strings
        'possible_fen_after': poss_fen_after_list,  # list of max_possible FEN strings
        # Numeric arrays (small)
        'possible_scalars': np.stack(poss_scalars_list),   # (M, 11)
        'possible_mask': possible_mask,                     # (M,)
        'tabular': tabular,                                 # (18,)
        'actual_idx': np.int64(actual_idx),
        'is_mistake': np.float32(is_mistake),
        'win_prob_before': wdl_before,                      # (3,)
        'win_prob_after': wdl_after,                        # (3,)
        'time_spent_log': time_spent_log,
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset — on-the-fly plane construction
# ---------------------------------------------------------------------------

class MIMOCompactDataset(Dataset):
    """
    Loads a compact .npz + string arrays and builds 47×8×8 planes on-the-fly
    from FEN in __getitem__.

    Expected .npz keys:
        fen_before (N,) object array of FEN strings
        game_to_position (N,) object array of GTP strings
        possible_uci (N, M) object array of UCI strings
        possible_fen_after (N, M) object array of FEN strings
        possible_scalars (N, M, 11) float32
        possible_mask (N, M) float32
        tabular (N, 18) float32
        actual_idx (N,) int64
        is_mistake (N,) float32
        win_prob_before (N, 3) float32
        win_prob_after (N, 3) float32
        time_spent_log (N,) float32
    """

    def __init__(self, npz_path: str, max_possible: int = 40):
        print(f"[DATA] Loading {npz_path} …")
        data = np.load(npz_path, allow_pickle=True)
        self.fen_before       = data['fen_before']           # (N,) object
        self.game_to_position = data['game_to_position']     # (N,) object
        self.possible_uci     = data['possible_uci']         # (N, M) object
        self.possible_fen_after = data['possible_fen_after'] # (N, M) object
        self.possible_scalars = data['possible_scalars']     # (N, M, 11)
        self.possible_mask    = data['possible_mask']        # (N, M)
        self.tabular          = data['tabular']              # (N, 18)
        self.actual_idx       = data['actual_idx']           # (N,)
        self.is_mistake       = data['is_mistake']           # (N,)
        self.win_prob_before  = data['win_prob_before']      # (N, 3)
        self.win_prob_after   = data['win_prob_after']       # (N, 3)
        self.time_spent_log   = data['time_spent_log']       # (N,)
        self.max_possible     = max_possible
        self.n = len(self.fen_before)
        print(f"[DATA] {self.n:,} examples loaded (compact, planes built on-the-fly)")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        fen = str(self.fen_before[idx])
        gtp = str(self.game_to_position[idx])

        # Build current position planes from FEN
        try:
            board = chess.Board(fen)
            history = parse_game_to_position(gtp)
            recent = history[-2:] if history else []
            current_planes = board_to_planes(board, recent)
        except Exception:
            current_planes = np.zeros((47, 8, 8), dtype=np.float32)
            board = None
            recent = []

        # Build planes for each candidate move's resulting position
        poss_planes = np.zeros((self.max_possible, 47, 8, 8), dtype=np.float32)
        for i in range(self.max_possible):
            if self.possible_mask[idx, i] < 0.5:
                break
            fen_after = str(self.possible_fen_after[idx, i])
            uci = str(self.possible_uci[idx, i])
            if not fen_after:
                continue
            try:
                board_after = chess.Board(fen_after)
                from_sq = chess.parse_square(uci[:2])
                to_sq = chess.parse_square(uci[2:4])
                poss_hist = [(from_sq, to_sq)] + (recent[:1] if recent else [])
                poss_planes[i] = board_to_planes(board_after, poss_hist)
            except Exception:
                pass

        return {
            'current_planes':  torch.from_numpy(current_planes).float(),
            'possible_planes': torch.from_numpy(poss_planes).float(),
            'possible_scalars': torch.from_numpy(self.possible_scalars[idx]).float(),
            'possible_mask':   torch.from_numpy(self.possible_mask[idx]).float(),
            'tabular':         torch.from_numpy(self.tabular[idx]).float(),
            'actual_idx':      torch.tensor(self.actual_idx[idx], dtype=torch.long),
            'is_mistake':      torch.tensor(self.is_mistake[idx], dtype=torch.float32),
            'win_prob_before': torch.from_numpy(self.win_prob_before[idx]).float(),
            'win_prob_after':  torch.from_numpy(self.win_prob_after[idx]).float(),
            'time_spent_log':  torch.tensor(self.time_spent_log[idx], dtype=torch.float32),
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
        parquet_files = sorted(root.rglob(f'{table_name}/*.parquet'))
        if not parquet_files:
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

    # ---- Split by game_id ----
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

    # ---- Build compact examples per split ----
    for split_name, split_gids in [('train', train_games), ('val', val_games), ('test', test_games)]:
        split_moves = moves_df[moves_df['game_id'].isin(split_gids)]
        print(f"\n[4/5] Building {split_name}: {len(split_moves):,} moves …")

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

        results: List[Optional[Dict]] = []
        if workers > 1:
            with Pool(workers) as pool:
                results = pool.map(build_one_compact, tasks, chunksize=64)
        else:
            results = [build_one_compact(t) for t in tasks]

        examples = [r for r in results if r is not None]
        print(f"       Built {len(examples):,} valid examples (dropped {len(tasks) - len(examples):,})")

        if not examples:
            print(f"       ⚠ Skipping {split_name} — no valid examples")
            continue

        # Stack compact arrays + string arrays
        npz_path = out / f'{split_name}.npz'
        np.savez_compressed(
            npz_path,
            # String data (object arrays)
            fen_before         = np.array([e['fen_before'] for e in examples], dtype=object),
            game_to_position   = np.array([e['game_to_position'] for e in examples], dtype=object),
            possible_uci       = np.array([e['possible_uci'] for e in examples], dtype=object),
            possible_fen_after = np.array([e['possible_fen_after'] for e in examples], dtype=object),
            # Numeric data
            possible_scalars   = np.stack([e['possible_scalars'] for e in examples]),
            possible_mask      = np.stack([e['possible_mask'] for e in examples]),
            tabular            = np.stack([e['tabular'] for e in examples]),
            actual_idx         = np.array([e['actual_idx'] for e in examples]),
            is_mistake         = np.array([e['is_mistake'] for e in examples]),
            win_prob_before    = np.stack([e['win_prob_before'] for e in examples]),
            win_prob_after     = np.stack([e['win_prob_after'] for e in examples]),
            time_spent_log     = np.array([e['time_spent_log'] for e in examples]),
        )
        sz_mb = os.path.getsize(npz_path) / (1024 * 1024)
        print(f"       Saved {npz_path}  ({sz_mb:.1f} MB)")

    # ---- Save config ----
    config = {
        'format': 'compact_v1',
        'max_possible': max_possible,
        'min_elo': min_elo,
        'max_elo': max_elo,
        'tabular_dim': 18,
        'move_scalar_dim': 11,
        'planes_dim': '47x8x8 (built on-the-fly from FEN)',
        'tabular_features': [
            'time_remaining/3600', 'white_elo/3000', 'black_elo/3000',
            'elo_diff/1000', 'move_no/200', 'color_01',
            'eval_stm_before/1000', 'stm_win_before', 'draw_perc_before',
            'stm_loss_before', 'initial_time/3600', 'increment/60',
            'prev_move_was_capture', 'in_check',
            'eval_std/1000', 'num_captures_frac', 'num_checks_frac',
            'num_candidates_frac',
        ],
        'move_scalar_features': [
            'eval_stm/1000', 'stm_win_perc', 'draw_perc', 'stm_loss_perc',
            'log1p(nodes)/20', 'depth/40', 'move_quality',
            'piece_type', 'is_capture', 'is_check', 'is_checkmate',
        ],
        'splits': {'val_frac': val_frac, 'test_frac': test_frac, 'seed': seed},
    }
    with open(out / 'dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n[5/5] Saved dataset_config.json")
    print("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build MIMO chess dataset (compact)')
    parser.add_argument('--data-dir',     required=True,
                        help='Root output dir (e.g. output/lc0/791556).')
    parser.add_argument('--output-dir',   required=True, help='Output directory for .npz splits')
    parser.add_argument('--max-possible', type=int, default=40)
    parser.add_argument('--min-elo',      type=int, default=0)
    parser.add_argument('--max-elo',      type=int, default=0)
    parser.add_argument('--val-frac',     type=float, default=0.10)
    parser.add_argument('--test-frac',    type=float, default=0.05)
    parser.add_argument('--workers',      type=int, default=4)
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    build_dataset(
        args.data_dir, args.output_dir,
        max_possible=args.max_possible,
        min_elo=args.min_elo, max_elo=args.max_elo,
        val_frac=args.val_frac, test_frac=args.test_frac,
        workers=args.workers, seed=args.seed,
    )
