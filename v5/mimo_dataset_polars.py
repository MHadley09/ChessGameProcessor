#!/usr/bin/env python3
"""
mimo_dataset_polars.py — Compact dataset for the MIMO chess model (per-worker streaming, chunked processing).

Reads from Parquet files produced by LC0 processing pipeline.
Processes one worker directory at a time with chunked task generation to avoid OOM.
Stores compact data: FENs, UCI moves, scalars, labels. Planes built on-the-fly.

This version processes moves in chunks of 100K to avoid building giant task lists.
"""

import argparse
import bisect
import json
import math
import os
import gc
import sys
import traceback
from collections import defaultdict, OrderedDict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force line buffering for immediate output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

import chess
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import polars as pl
from torch.utils.data import Dataset, Sampler

print("Imports complete, starting...", flush=True)


# ---------------------------------------------------------------------------
# Plane codec
# ---------------------------------------------------------------------------

def _bb_to_planes_batch(bitboards: List[int]) -> np.ndarray:
    """Convert N chess bitboards → (N, 8, 8) float32 planes via numpy batch ops.
    
    ~5-7× faster than per-square Python iteration. Each bitboard is a uint64
    from python-chess (e.g. int(board.pieces(PAWN, WHITE))).
    """
    n = len(bitboards)
    if n == 0:
        return np.zeros((0, 8, 8), dtype=np.float32)
    arr = np.array(bitboards, dtype=np.uint64)
    raw = arr.view(np.uint8).reshape(n, 8)              # N × 8 bytes (little-endian)
    bits = np.unpackbits(raw, axis=1).reshape(n, 8, 8)  # MSB-first per byte
    bits = np.flip(bits, axis=2)                         # fix file order (LSB = A-file)
    bits = np.flip(bits, axis=1)                         # rank 7 at row 0 (board top)
    return np.ascontiguousarray(bits).astype(np.float32)


def _piece_bitboards(board: chess.Board) -> List[int]:
    """Return 12 bitboards in plane order: [W-pawn..W-king, B-pawn..B-king]."""
    bbs = []
    for pt in range(1, 7):
        bbs.append(int(board.pieces(pt, chess.WHITE)))
    for pt in range(1, 7):
        bbs.append(int(board.pieces(pt, chess.BLACK)))
    return bbs


NUM_PLANES = 23  # Reduced from 47 — history planes were always zeros (board from FEN has no move stack)

def board_to_planes(board: chess.Board, history: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    """Bitboard-accelerated FEN → (23, 8, 8) plane construction.

    Plane layout (23 channels):
        0-11:  piece planes (W-pawn..W-king, B-pawn..B-king)
        12-13: last move from/to squares
        14-15: second-to-last move from/to squares
        16:    side to move (1.0 = White)
        17:    White kingside castling
        18:    White queenside castling
        19:    Black kingside castling
        20:    Black queenside castling
        21:    en passant square
        22:    halfmove clock / 100

    History piece planes (old 12-35) removed — the board is constructed
    from fen_before which has no move stack, so those channels were always
    zeros.  Move history is still encoded via from/to square planes 12-15.
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    # --- Piece planes 0-11 ---
    bbs = _piece_bitboards(board)
    p12 = _bb_to_planes_batch(bbs)
    for i in range(12):
        if bbs[i]:
            planes[i] = p12[i]

    # --- Move history squares 12-15 ---
    if history:
        if len(history) >= 1:
            fr, to = history[0]
            if fr is not None:
                planes[12, 7 - fr // 8, fr % 8] = 1.0
                planes[13, 7 - to // 8, to % 8] = 1.0
        if len(history) >= 2:
            fr, to = history[1]
            if fr is not None:
                planes[14, 7 - fr // 8, fr % 8] = 1.0
                planes[15, 7 - to // 8, to % 8] = 1.0

    # --- Metadata planes 16-22 ---
    planes[16, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[17, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[18, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[19, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[20, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    if board.ep_square is not None:
        planes[21, 7 - board.ep_square // 8, board.ep_square % 8] = 1.0
    planes[22, :, :] = min(board.halfmove_clock, 100) / 100.0
    return planes


# ---------------------------------------------------------------------------
# UCI move → (from_sq, to_sq, promo) parser
# ---------------------------------------------------------------------------

def parse_uci_move(uci: str) -> tuple:
    """
    Parse a UCI move string into (from_sq, to_sq, promo) integers.

    from_sq, to_sq: 0-63 chess square index (0=a1, 63=h8)
    promo: 0=none, 1=knight, 2=bishop, 3=rook, 4=queen

    Examples:
        parse_uci_move("e2e4")  → (12, 28, 0)
        parse_uci_move("e7e8q") → (52, 60, 4)
    """
    from_file = ord(uci[0]) - ord('a')
    from_rank = int(uci[1]) - 1
    to_file   = ord(uci[2]) - ord('a')
    to_rank   = int(uci[3]) - 1
    from_sq = from_rank * 8 + from_file
    to_sq   = to_rank   * 8 + to_file
    promo = 0
    if len(uci) >= 5:
        promo = {'n': 1, 'b': 2, 'r': 3, 'q': 4}.get(uci[4].lower(), 0)
    return from_sq, to_sq, promo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_game_to_position(gtp_str: str) -> List[Tuple[int, int]]:
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


_WDL_WHITE_WIN = np.array([1., 0., 0.], dtype=np.float32)
_WDL_DRAW      = np.array([0., 1., 0.], dtype=np.float32)
_WDL_BLACK_WIN = np.array([0., 0., 1.], dtype=np.float32)

def result_to_wdl(result: str) -> np.ndarray:
    """Game outcome from White's perspective. No color parameter needed."""
    if result == '1-0':
        return _WDL_WHITE_WIN.copy()
    elif result == '0-1':
        return _WDL_BLACK_WIN.copy()
    else:
        return _WDL_DRAW.copy()


def _sanitize_np(x, threshold: float = 1000.0, bound: float = 50.0):
    """Map non-finite + sentinel-magnitude entries to neutral 0.0, then bound.

    Mirrors sanitize_features() in chess_mimo_model_v5.py so stored-shard
    sentinels (e.g. -42069) are cleaned at source. Behavior-neutral for
    in-distribution features (|value| << threshold).
    """
    import numpy as _np
    x = _np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = _np.where(_np.abs(x) > threshold, 0.0, x)
    return _np.clip(x, -bound, bound).astype(_np.float32)


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        v = float(val)
        if v != v:  # NaN check without math.isnan overhead
            return default
        return v
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def parse_time_control(tc) -> Tuple[float, float]:
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
    try:
        return float(tc), 0.0
    except ValueError:
        return 0.0, 0.0


def detect_prev_capture(game_to_position_str: str) -> float:
    """Check if the last move leading to this position was a capture.
    
    Replays game_to_position moves to determine if the final move captured.
    Still requires replay since we need the board state before the last move.
    """
    if not game_to_position_str or (isinstance(game_to_position_str, float) and math.isnan(game_to_position_str)):
        return 0.0
    tokens = str(game_to_position_str).split(',')
    tokens = [t.strip() for t in tokens if len(t.strip()) >= 4]
    if not tokens:
        return 0.0
    try:
        replay = chess.Board()
        for tok in tokens[:-1]:
            from_sq = chess.parse_square(tok[:2])
            to_sq = chess.parse_square(tok[2:4])
            for legal in replay.legal_moves:
                if legal.from_square == from_sq and legal.to_square == to_sq:
                    replay.push(legal)
                    break
        last = tokens[-1]
        from_sq = chess.parse_square(last[:2])
        to_sq = chess.parse_square(last[2:4])
        # Direct check: piece on destination = capture (or en passant)
        if replay.piece_at(to_sq) is not None:
            return 1.0
        # En passant
        if replay.ep_square == to_sq:
            piece = replay.piece_at(from_sq)
            if piece and piece.piece_type == chess.PAWN:
                return 1.0
        return 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Build one example (top-level for pickling)
# ---------------------------------------------------------------------------


def compute_game_phase_from_board(board, ply: int) -> int:
    """Heuristic game phase anchor for PhaseEncoder / FiLM conditioning.

    These are NOT ground-truth labels — they provide initial anchoring so
    the PhaseEncoder and FiLM conditioner have a reasonable starting signal.
    The model learns the real phase boundaries through training.

    Anchoring rules:
        Opening (0):    ply <= 20 (~first 10 full moves)
        Endgame (2):    material heuristic — no queens and each side has
                        at most rook + minor, or queen(s) with no other pieces
        Middlegame (1): everything else

    Args:
        board: chess.Board instance
        ply: half-move count from game start
    """
    # --- Opening: first ~10 full moves ---
    if ply <= 20:
        return 0

    # --- Endgame: material-based ---
    import chess
    w_queen = len(board.pieces(chess.QUEEN, chess.WHITE))
    b_queen = len(board.pieces(chess.QUEEN, chess.BLACK))
    w_rook  = len(board.pieces(chess.ROOK, chess.WHITE))
    b_rook  = len(board.pieces(chess.ROOK, chess.BLACK))
    w_minor = (len(board.pieces(chess.BISHOP, chess.WHITE))
             + len(board.pieces(chess.KNIGHT, chess.WHITE)))
    b_minor = (len(board.pieces(chess.BISHOP, chess.BLACK))
             + len(board.pieces(chess.KNIGHT, chess.BLACK)))

    # No queens: endgame if each side has at most a rook + minor piece
    if w_queen == 0 and b_queen == 0:
        if (w_rook + w_minor) <= 2 and (b_rook + b_minor) <= 2:
            return 2

    # One or both sides have queen(s): endgame only if the queen is
    # the sole non-pawn piece for that side
    else:
        w_others = w_rook + w_minor  # non-queen, non-pawn, non-king
        b_others = b_rook + b_minor
        if w_others == 0 and b_others == 0:
            return 2

    # --- Middlegame: everything else ---
    return 1

def build_one_compact(move_dict: Dict, game_dict: Dict, possibles: List[Dict], max_possible: int, with_phase: bool = True) -> Optional[Dict[str, Any]]:
    fen_before = move_dict.get('fen_before', '')
    try:
        board = chess.Board(fen_before)
    except Exception:
        return None
    if not possibles:
        return None

    color = move_dict.get('color', 'White')
    # Exclude moves by BOT players
    title = str(game_dict.get('white_title' if color == 'White' else 'black_title', '')).strip()
    if title.upper() == 'BOT':
        return None
    game_result = game_dict.get('result', '1/2-1/2')
    w_elo = _safe_float(game_dict.get('white_elo'), 1500)
    b_elo = _safe_float(game_dict.get('black_elo'), 1500)

    possibles = sorted(possibles, key=lambda x: _safe_float(x.get('eval'), -99999), reverse=True)
    possibles = possibles[:max_possible]
    num_possible = len(possibles)

    is_white = color == 'White'
    sign = 1 if is_white else -1

    evals_stm = []
    for pm in possibles:
        evals_stm.append(_safe_float(pm.get('eval'), 0) * sign)
    best_eval_stm = max(evals_stm) if evals_stm else 0.0
    worst_eval_stm = min(evals_stm) if evals_stm else 0.0
    eval_range = best_eval_stm - worst_eval_stm
    inv_eval_range = 1.0 / eval_range if eval_range > 0 else 0.0

    # Pre-compute check/checkmate by pushing moves on the board (avoids N Board constructions)
    legal_move_info = {}
    for move in board.legal_moves:
        uci = move.uci()
        board.push(move)
        legal_move_info[uci] = (board.is_check(), board.is_checkmate())
        board.pop()

    poss_scalars = np.zeros((max_possible, 13), dtype=np.float32)
    poss_uci_list = []
    poss_fen_after_list = []
    piece_map = {'P': 1/6, 'N': 2/6, 'B': 3/6, 'R': 4/6, 'Q': 5/6, 'K': 1.0}

    actual_uci = move_dict.get('move', '')
    actual_idx = -1

    for i, pm in enumerate(possibles):
        uci = pm.get('move', '')
        poss_uci_list.append(uci)
        poss_fen_after_list.append(pm.get('fen_after', ''))
        if uci == actual_uci:
            actual_idx = i

        pm_eval_stm = evals_stm[i]
        w_win  = _safe_float(pm.get('white_win_perc'), 0.33)
        w_draw = _safe_float(pm.get('draw_perc'), 0.34)
        w_loss = _safe_float(pm.get('black_win_perc'), 0.33)
        nodes_raw = _safe_float(pm.get('nodes'), 1)
        move_quality = (pm_eval_stm - worst_eval_stm) * inv_eval_range if eval_range > 0 else 1.0
        piece_val = piece_map.get(str(pm.get('piece', 'P')).upper(), 1/6)

        # Capture detection — use board directly
        try:
            to_sq_int = chess.parse_square(pm['to_square'])
            is_capture = 1.0 if board.piece_at(to_sq_int) is not None else 0.0
            if board.ep_square == to_sq_int and str(pm.get('piece', '')).upper() == 'P':
                is_capture = 1.0
        except Exception:
            is_capture = 0.0

        # Check/checkmate from pre-computed dict (no Board construction per move)
        chk_info = legal_move_info.get(uci)
        if chk_info is not None:
            is_check = 1.0 if chk_info[0] else 0.0
            is_checkmate = 1.0 if chk_info[1] else 0.0
        else:
            is_check = 0.0
            is_checkmate = 0.0

        poss_scalars[i] = [
            pm_eval_stm / 1000.0,
            w_win,
            w_draw,
            w_loss,
            math.log1p(nodes_raw) / 20.0,
            _safe_float(pm.get('depth'), 20) / 40.0,
            move_quality,
            piece_val,
            is_capture,
            is_check,
            is_checkmate,
            0.0,  # is_mistake_move — filled after EV computation
            0.0,  # is_excellent_move — filled after EV computation
        ]

    # Pad UCI/FEN lists
    pad_count = max_possible - num_possible
    if pad_count > 0:
        poss_uci_list.extend([''] * pad_count)
        poss_fen_after_list.extend([''] * pad_count)

    possible_mask = np.zeros(max_possible, dtype=np.float32)
    possible_mask[:num_possible] = 1.0

    eval_raw = _safe_float(move_dict.get('eval_before'), 0)
    eval_stm = eval_raw * sign

    # WDL before — always White's perspective
    white_win_before  = _safe_float(move_dict.get('white_win_perc_before'), 0.33)
    white_draw_before = _safe_float(move_dict.get('draw_perc_before'), 0.34)
    white_loss_before = _safe_float(move_dict.get('black_win_perc_before'), 0.33)

    initial_time, increment = parse_time_control(game_dict.get('time_control', ''))
    prev_capture = move_dict.get('_prev_capture', 0.0)
    in_check = 1.0 if board.is_check() else 0.0
    eval_std = float(np.std(evals_stm)) / 1000.0 if len(evals_stm) > 1 else 0.0

    # Aggregate from pre-built scalars instead of iterating lists again
    if num_possible > 0:
        num_captures = float(poss_scalars[:num_possible, 8].sum()) / num_possible
        num_checks = float(poss_scalars[:num_possible, 9].sum()) / num_possible
    else:
        num_captures = 0.0
        num_checks = 0.0
    num_candidates = num_possible / max_possible

    # ------------------------------------------------------------------
    # STM Expected Value + per-move mistake/excellent flags
    # EV = win * 1.0 + draw * 0.5 + loss * 0.0 (from side-to-move perspective)
    # WDL in poss_scalars slots 1-3 are white_win%, draw%, black_win% on 0-100
    # ------------------------------------------------------------------
    is_mistake = 0.0
    frac_mistake_moves = 0.0
    frac_excellent_moves = 0.0

    if num_possible > 0:
        w_pcts = poss_scalars[:num_possible, 1] / 100.0
        d_pcts = poss_scalars[:num_possible, 2] / 100.0
        l_pcts = poss_scalars[:num_possible, 3] / 100.0
        if is_white:
            stm_evs = w_pcts + 0.5 * d_pcts
        else:
            stm_evs = l_pcts + 0.5 * d_pcts
        best_idx = int(np.argmax(stm_evs))
        best_ev = stm_evs[best_idx]
        drops = best_ev - stm_evs
        for j in range(num_possible):
            poss_scalars[j, 11] = 1.0 if drops[j] > 0.25 else 0.0
            poss_scalars[j, 12] = 1.0 if (drops[j] < 0.025 or j == best_idx) else 0.0
        frac_mistake_moves = float(poss_scalars[:num_possible, 11].sum()) / num_possible
        frac_excellent_moves = float(poss_scalars[:num_possible, 12].sum()) / num_possible
        if actual_idx >= 0:
            is_mistake = 1.0 if drops[actual_idx] > 0.25 else 0.0

    tabular = np.array([
        _safe_float(move_dict.get('time_remaining')) / 3600.0,
        w_elo / 3000.0,
        b_elo / 3000.0,
        (w_elo - b_elo) / 1000.0,
        _safe_float(move_dict.get('move_no')) / 200.0,
        1.0 if is_white else 0.0,
        eval_stm / 1000.0,
        white_win_before,
        white_draw_before,
        white_loss_before,
        initial_time / 3600.0,
        increment / 60.0,
        prev_capture,
        in_check,
        eval_std,
        num_captures,
        num_checks,
        num_candidates,
        frac_mistake_moves,
        frac_excellent_moves,
    ], dtype=np.float32)

    wdl_before = result_to_wdl(game_result)
    raw_ts = max(0.0, _safe_float(move_dict.get('time_spent')))
    time_spent_log = np.float32(math.log1p(raw_ts))
    gtp = str(move_dict.get('game_to_position', '')) if move_dict.get('game_to_position') else ''

    result = {
        'fen_before': fen_before,
        'game_to_position': gtp,
        'possible_uci': poss_uci_list,
        'possible_fen_after': poss_fen_after_list,
        'possible_scalars': poss_scalars,
        'possible_mask': possible_mask,
        'tabular': tabular,
        'actual_idx': np.int64(actual_idx),
        'is_mistake': np.float32(is_mistake),
        'win_prob_before': wdl_before,
        'time_spent_log': time_spent_log,
    }
    
    # Phase anchor for PhaseEncoder/FiLM (opening=ply, endgame=material, else=midgame)
    try:
        ply = int(_safe_float(move_dict.get('move_no'), 0))
    except Exception:
        ply = 0
    try:
        _fen = move_dict.get('fen_before', '')
        _phase_board = chess.Board(_fen) if _fen else chess.Board()
        result['game_phase'] = np.int64(compute_game_phase_from_board(_phase_board, ply))
    except Exception:
        result['game_phase'] = np.int64(1)  # default middlegame
    
    return result


# Wrapper for executor (picklable)
def build_one_compact_wrapper(args_tuple):
    return build_one_compact(*args_tuple)


# ---------------------------------------------------------------------------
# Shard-aware sampler — eliminates cross-shard cache thrashing
# ---------------------------------------------------------------------------

class ShardGroupSampler(Sampler):
    """Shuffle shards, then shuffle within each shard.

    Standard random shuffle scatters indices across all shards, causing
    constant cache misses in the DataLoader workers.  This sampler groups
    indices by shard so each worker processes one shard at a time, giving
    near-100% cache hits while still randomising across epochs.
    """

    def __init__(self, shard_offsets, shard_counts, seed=42):
        self.shard_offsets = list(shard_offsets)
        self.shard_counts = list(shard_counts)
        self.total = sum(shard_counts)
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        shard_order = torch.randperm(len(self.shard_counts), generator=g)
        for s in shard_order:
            offset = self.shard_offsets[s]
            count = self.shard_counts[s]
            local_perm = torch.randperm(count, generator=g)
            yield from (offset + local_perm).tolist()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.total


# ---------------------------------------------------------------------------
# V5 on-the-fly scalar enrichment
# ---------------------------------------------------------------------------

def _enrich_v5_scalars(possible_scalars, possible_mask, tabular):
    """Compute V5 per-move is_mistake (slot 11) and is_excellent (slot 12)
    from WDL already in slots 1-3.  Also refreshes tabular slots 18-19
    (frac_mistake_moves, frac_excellent_moves) so they stay consistent
    with the V5 thresholds regardless of what the writer stored.

    Operates in-place.  Cost: vectorised numpy on ≤220 elements — negligible.
    """
    n = int(possible_mask.sum())
    if n == 0:
        return
    # WDL in slots 1-3 are white_win%, draw%, black_win% on 0-100 scale
    w_pct = possible_scalars[:n, 1] / 100.0
    d_pct = possible_scalars[:n, 2] / 100.0
    l_pct = possible_scalars[:n, 3] / 100.0
    # Side-to-move: tabular[5] == 1.0 → white
    if tabular[5] > 0.5:
        stm_ev = w_pct + 0.5 * d_pct
    else:
        stm_ev = l_pct + 0.5 * d_pct
    best_ev = stm_ev.max()
    drops = best_ev - stm_ev
    # Slot 11 — per-move is_mistake  (EV drop > 0.25 from best)
    possible_scalars[:n, 11] = (drops > 0.25).astype(np.float32)
    # Slot 12 — per-move is_excellent (drop < 0.025 OR is best move)
    excellent = drops < 0.025  # includes best_idx automatically
    possible_scalars[:n, 12] = excellent.astype(np.float32)
    # Zero beyond valid moves (already zero from allocation, but be safe)
    possible_scalars[n:, 11:13] = 0.0
    # Refresh tabular frac features (slots 18-19) to match V5 thresholds
    tabular[18] = possible_scalars[:n, 11].sum() / n
    tabular[19] = possible_scalars[:n, 12].sum() / n


# ---------------------------------------------------------------------------
# Sharded Dataset
# ---------------------------------------------------------------------------

class MIMOCompactDataset(Dataset):
    # Keys that __getitem__ actually uses (possible_fen_after dropped — push/pop is faster)
    _SHARD_LOAD_KEYS = frozenset([
        'fen_before', 'game_to_position', 'possible_uci',
        'possible_mask',
        'tabular',
        'actual_idx', 'is_mistake', 'win_prob_before',
        'time_spent_log',
    ])
    # Numeric arrays that can be memory-mapped (zero per-worker RAM)
    # NOTE: possible_scalars is stored as sparse (scalars_data + scalars_offsets)
    # NOTE: possible_from_sq/to_sq/promo are parsed at runtime from possible_uci
    _NUMERIC_KEYS = frozenset([
        'possible_mask',
        'tabular',
        'actual_idx', 'is_mistake', 'win_prob_before',
        'time_spent_log',
    ])
    # Dtype overrides for .npy cache compression (4.2x disk reduction).
    # All arrays are cast back to original dtypes in __getitem__.
    _DTYPE_OVERRIDES = {
        'current_planes': np.float16,
        'possible_mask': np.bool_,
        'actual_idx': np.int16,
    }
    # Object arrays that must be fully loaded (small, ~175 MB per shard)
    _OBJECT_KEYS = frozenset(['fen_before', 'game_to_position', 'possible_uci'])

    def __init__(self, data_path: str, max_possible: int = 220, cache_shards: int = 2,
                 with_phase: bool = True, no_npy_cache: bool = False):
        self.data_path = Path(data_path)
        self.max_possible = max_possible
        self.cache_shards = cache_shards
        self.with_phase = with_phase
        self.no_npy_cache = no_npy_cache

        if self.data_path.is_dir():
            self.shard_files = sorted([str(f) for f in self.data_path.glob('*.npz')])
            if not self.shard_files:
                raise FileNotFoundError(f"No .npz shards in {data_path}")

            if self.no_npy_cache:
                # --- Direct npz mode: load shards into RAM, no disk cache ---
                print(f"[DATA] Direct npz mode (no .npy cache). "
                      f"{len(self.shard_files)} shards, ~{self.cache_shards} cached in RAM.",
                      flush=True)
                self._shard_npy_dirs = None  # Not used in direct mode

                # Count shard sizes by peeking at actual_idx in each npz
                self.shard_offsets = []
                self.shard_counts = []
                total = 0
                for i, f in enumerate(self.shard_files):
                    npz = np.load(f, allow_pickle=True)
                    n = len(npz['actual_idx'])
                    npz.close()
                    self.shard_offsets.append(total)
                    self.shard_counts.append(n)
                    total += n
                self.n = total
                self._shard_cache = OrderedDict()
            else:
                # --- Standard mode: one-time extraction npz → .npy for mmap ---
                self._shard_npy_dirs = []
                npy_cache_root = self.data_path / '.npy_cache'
                needs_extraction = False
                for f in self.shard_files:
                    npy_dir = npy_cache_root / Path(f).stem
                    self._shard_npy_dirs.append(npy_dir)
                    if not (npy_dir / '.ready').exists():
                        needs_extraction = True

                if needs_extraction:
                    npy_cache_root.mkdir(parents=True, exist_ok=True)
                    print("[DATA] Extracting shards to .npy cache for memory-mapped access "
                          "(one-time, subsequent runs will skip)...", flush=True)
                    for i, f in enumerate(self.shard_files):
                        npy_dir = self._shard_npy_dirs[i]
                        if (npy_dir / '.ready').exists():
                            continue
                        npy_dir.mkdir(parents=True, exist_ok=True)
                        stem = Path(f).stem
                        print(f"  [{i+1}/{len(self.shard_files)}] {stem}...", end='', flush=True)
                        npz = np.load(f, allow_pickle=True)
                        for k in npz.files:
                            # Only extract numeric keys (mmap-able). Object arrays
                            # (fen_before, game_to_position, possible_uci) are loaded
                            # from the original .npz at shard-load time — saves ~30-40%
                            # cache disk space.
                            if k == 'possible_scalars':
                                # Phase 2: sparse CSR-like storage for possible_scalars
                                # Only store real (non-padded) moves — 86% less disk
                                scalars = npz['possible_scalars']     # (shard_N, max_possible, 13)
                                mask = npz['possible_mask']           # (shard_N, max_possible)
                                n_legal = mask.sum(axis=1).astype(np.int32)
                                offsets = np.zeros(len(n_legal) + 1, dtype=np.int32)
                                np.cumsum(n_legal, out=offsets[1:])
                                total_moves = int(offsets[-1])
                                scalar_dim = scalars.shape[2] if scalars.ndim == 3 else 13
                                data = np.zeros((total_moves, scalar_dim), dtype=np.float32)
                                for row_i in range(len(scalars)):
                                    data[offsets[row_i]:offsets[row_i + 1]] = scalars[row_i, :n_legal[row_i]]
                                np.save(str(npy_dir / 'scalars_data.npy'), data)
                                np.save(str(npy_dir / 'scalars_offsets.npy'), offsets)
                            elif k in self._NUMERIC_KEYS or k == 'game_phase':
                                arr = npz[k]
                                if k in self._DTYPE_OVERRIDES:
                                    arr = arr.astype(self._DTYPE_OVERRIDES[k])
                                np.save(str(npy_dir / f'{k}.npy'), arr, allow_pickle=True)
                        npz.close()
                        del npz
                        gc.collect()
                        (npy_dir / '.ready').touch()
                        print(" done", flush=True)
                    print("[DATA] Extraction complete.", flush=True)

                # Count shard sizes from the lightweight actual_idx .npy (fastest to load)
                self.shard_offsets = []
                self.shard_counts = []
                total = 0
                for npy_dir in self._shard_npy_dirs:
                    count_arr = np.load(str(npy_dir / 'actual_idx.npy'), mmap_mode='r')
                    n = len(count_arr)
                    self.shard_offsets.append(total)
                    self.shard_counts.append(n)
                    total += n
                    del count_arr
                self.n = total
                self._shard_cache = OrderedDict()
        else:
            data = np.load(self.data_path, allow_pickle=True)
            self.fen_before = data['fen_before']
            self.game_to_position = data['game_to_position']
            self.possible_uci = data['possible_uci']
            self.possible_scalars = data['possible_scalars']
            self.possible_mask = data['possible_mask']
            self.tabular = data['tabular']
            self.actual_idx = data['actual_idx']
            self.is_mistake = data['is_mistake']
            self.win_prob_before = data['win_prob_before']
            self.time_spent_log = data['time_spent_log']
            if 'game_phase' in data:
                self.game_phase = data['game_phase']
            self.n = len(self.fen_before)
            self.shard_files = None
            self._shard_npy_dirs = None
        print(f"[DATA] {self.n:,} examples", flush=True)

    def __len__(self):
        return self.n

    def _load_shard(self, shard_idx):
        """Load shard data — direct from npz or via .npy cache depending on mode."""
        if shard_idx in self._shard_cache:
            self._shard_cache.move_to_end(shard_idx)
            return self._shard_cache[shard_idx]
        while len(self._shard_cache) >= self.cache_shards:
            self._shard_cache.popitem(last=False)

        if self.no_npy_cache:
            return self._load_shard_direct(shard_idx)
        else:
            return self._load_shard_npy(shard_idx)

    def _load_shard_direct(self, shard_idx):
        """Load shard directly from npz into RAM (no disk cache).

        All arrays are decompressed into memory. CSR conversion for
        possible_scalars is done in-memory. With ShardGroupSampler and
        cache_shards=3, peak RAM is ~2-3 GB for 250 MB compressed shards.
        """
        shard_data = {}
        npz = np.load(self.shard_files[shard_idx], allow_pickle=True)

        # Build CSR for possible_scalars in memory
        if 'possible_scalars' in npz.files:
            scalars = npz['possible_scalars']
            mask = npz['possible_mask']
            n_legal = mask.sum(axis=1).astype(np.int32)
            offsets = np.zeros(len(n_legal) + 1, dtype=np.int32)
            np.cumsum(n_legal, out=offsets[1:])
            total_moves = int(offsets[-1])
            scalar_dim = scalars.shape[2] if scalars.ndim == 3 else 13
            data = np.zeros((total_moves, scalar_dim), dtype=np.float32)
            for row_i in range(len(scalars)):
                data[offsets[row_i]:offsets[row_i + 1]] = scalars[row_i, :n_legal[row_i]]
            shard_data['scalars_data'] = data
            shard_data['scalars_offsets'] = offsets
            del scalars  # free the dense array

        # Numeric arrays: load fully into RAM with dtype overrides
        for k in self._NUMERIC_KEYS:
            if k in npz.files:
                arr = npz[k]
                if k in self._DTYPE_OVERRIDES:
                    arr = arr.astype(self._DTYPE_OVERRIDES[k])
                shard_data[k] = arr

        # Object arrays: load fully into RAM
        for k in self._OBJECT_KEYS:
            if k in npz.files:
                shard_data[k] = npz[k]

        # Game phase if present
        if 'game_phase' in npz.files:
            shard_data['game_phase'] = npz['game_phase']

        npz.close()
        del npz

        self._shard_cache[shard_idx] = shard_data
        return shard_data

    def _load_shard_npy(self, shard_idx):
        """Load shard data via memory-mapped .npy files.

        Numeric arrays (possible_scalars, etc.) are memory-mapped — the OS
        pages in only the rows actually accessed, so per-worker RAM is near
        zero.  Object arrays (fen strings, UCI strings) are small and loaded
        fully (~175 MB per shard).
        """
        npy_dir = self._shard_npy_dirs[shard_idx]
        shard_data = {}

        # Numeric arrays: memory-mapped from .npy cache (zero per-worker RAM)
        for k in self._NUMERIC_KEYS:
            npy_path = npy_dir / f'{k}.npy'
            if npy_path.exists():
                shard_data[k] = np.load(str(npy_path), mmap_mode='r')

        # Sparse possible_scalars: mmap data, fully load small offsets
        sd_path = npy_dir / 'scalars_data.npy'
        so_path = npy_dir / 'scalars_offsets.npy'
        if sd_path.exists() and so_path.exists():
            shard_data['scalars_data'] = np.load(str(sd_path), mmap_mode='r')
            shard_data['scalars_offsets'] = np.load(str(so_path))  # small, load fully

        # Object arrays: load from original .npz (not extracted to save disk)
        npz = np.load(self.shard_files[shard_idx], allow_pickle=True)
        for k in self._OBJECT_KEYS:
            if k in npz.files:
                shard_data[k] = npz[k]
        # Keep npz handle alive so arrays stay valid; store ref for cleanup
        shard_data['_npz_handle'] = npz

        # Always load game_phase if saved; computed on-the-fly in __getitem__ if missing
        gp_path = npy_dir / 'game_phase.npy'
        if gp_path.exists():
            shard_data['game_phase'] = np.load(str(gp_path), mmap_mode='r')
        self._shard_cache[shard_idx] = shard_data
        return shard_data

    def __getitem__(self, idx):
        if self.shard_files:
            shard_idx = bisect.bisect_right(self.shard_offsets, idx) - 1
            local_idx = idx - self.shard_offsets[shard_idx]
            data = self._load_shard(shard_idx)
            fen_before = str(data['fen_before'][local_idx])
            gtp = str(data['game_to_position'][local_idx])
            possible_uci = data['possible_uci'][local_idx]
            # Reconstruct possible_scalars from sparse CSR storage
            if 'scalars_offsets' in data:
                s_start = int(data['scalars_offsets'][local_idx])
                s_end = int(data['scalars_offsets'][local_idx + 1])
                n_moves = s_end - s_start
                raw_dim = data['scalars_data'].shape[1] if 'scalars_data' in data else 13
                possible_scalars = np.zeros((self.max_possible, 13), dtype=np.float32)
                if n_moves > 0:
                    possible_scalars[:n_moves, :raw_dim] = data['scalars_data'][s_start:s_end]
            else:
                raw = data['possible_scalars'][local_idx]
                if raw.shape[-1] < 13:
                    possible_scalars = np.zeros((self.max_possible, 13), dtype=np.float32)
                    possible_scalars[:, :raw.shape[-1]] = raw
                else:
                    possible_scalars = raw
            possible_scalars = np.array(possible_scalars, dtype=np.float32)
            possible_mask = data['possible_mask'][local_idx]
            tabular = np.array(data['tabular'][local_idx], dtype=np.float32)
            # Pad tabular to 20 dims if shard was built with fewer (V4 = 18)
            if tabular.shape[-1] < 20:
                padded = np.zeros(20, dtype=np.float32)
                padded[:tabular.shape[-1]] = tabular
                tabular = padded
            _enrich_v5_scalars(possible_scalars, possible_mask, tabular)
            actual_idx = int(data['actual_idx'][local_idx])
            is_mistake = float(data['is_mistake'][local_idx])
            win_prob_before = data['win_prob_before'][local_idx]
            time_spent_log = float(data['time_spent_log'][local_idx])
            if 'game_phase' in data:
                game_phase = int(data['game_phase'][local_idx])
            else:
                # Compute from FEN if not saved in dataset
                try:
                    board = chess.Board(str(data['fen_before'][local_idx]))
                    ply = (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1)
                    game_phase = compute_game_phase_from_board(board, ply)
                except Exception:
                    game_phase = 1  # default middlegame
        else:
            fen_before = str(self.fen_before[idx])
            gtp = str(self.game_to_position[idx])
            possible_uci = self.possible_uci[idx]
            possible_scalars = np.array(self.possible_scalars[idx], dtype=np.float32)
            if possible_scalars.shape[-1] < 13:
                padded_ps = np.zeros((self.max_possible, 13), dtype=np.float32)
                padded_ps[:, :possible_scalars.shape[-1]] = possible_scalars
                possible_scalars = padded_ps
            possible_mask = self.possible_mask[idx]
            tabular = np.array(self.tabular[idx], dtype=np.float32)
            if tabular.shape[-1] < 20:
                padded_tab = np.zeros(20, dtype=np.float32)
                padded_tab[:tabular.shape[-1]] = tabular
                tabular = padded_tab
            _enrich_v5_scalars(possible_scalars, possible_mask, tabular)
            actual_idx = int(self.actual_idx[idx])
            is_mistake = float(self.is_mistake[idx])
            win_prob_before = self.win_prob_before[idx]
            time_spent_log = float(self.time_spent_log[idx])
            if hasattr(self, 'game_phase'):
                game_phase = int(self.game_phase[idx])
            else:
                try:
                    board = chess.Board(str(self.fen_before[idx]))
                    ply = (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1)
                    game_phase = compute_game_phase_from_board(board, ply)
                except Exception:
                    game_phase = 1  # default middlegame

        try:
            board = chess.Board(fen_before)
            history = parse_game_to_position(gtp)
            recent = history[-2:] if history else []
            current_planes = board_to_planes(board, recent)
        except Exception:
            current_planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
            recent = []

        # Parse from_sq/to_sq/promo from UCI move strings at runtime
        from_sqs = np.zeros(self.max_possible, dtype=np.int64)
        to_sqs   = np.zeros(self.max_possible, dtype=np.int64)
        promos   = np.zeros(self.max_possible, dtype=np.int64)
        for i in range(self.max_possible):
            if possible_mask[i] < 0.5:
                break
            uci = str(possible_uci[i]) if i < len(possible_uci) else ''
            if not uci or len(uci) < 4:
                continue
            try:
                f, t, p = parse_uci_move(uci)
                from_sqs[i] = f
                to_sqs[i]   = t
                promos[i]   = p
            except Exception:
                pass

        out = {
            'current_planes': torch.from_numpy(np.array(current_planes, dtype=np.float16)),
            'possible_scalars': torch.from_numpy(_sanitize_np(np.array(possible_scalars, dtype=np.float32))),
            'possible_mask': torch.from_numpy(np.array(possible_mask, dtype=np.float32)),
            'possible_from_sq': torch.from_numpy(from_sqs),
            'possible_to_sq':   torch.from_numpy(to_sqs),
            'possible_promo':   torch.from_numpy(promos),
            'tabular': torch.from_numpy(_sanitize_np(np.array(tabular, dtype=np.float32))),
            'actual_idx': torch.tensor(int(actual_idx), dtype=torch.long),
            'is_mistake': torch.tensor(is_mistake, dtype=torch.float32),
            'win_prob_before': torch.from_numpy(np.array(win_prob_before, dtype=np.float32)),
            'time_spent_log': torch.tensor(time_spent_log, dtype=torch.float32),
        }
        out['game_phase'] = torch.tensor(game_phase, dtype=torch.long)
        return out


# ---------------------------------------------------------------------------
# Process moves in chunks
# ---------------------------------------------------------------------------

def precompute_prev_captures(moves_df):
    """Replay each game once and store prev_was_capture for every move.
    
    Returns dict: (game_id, move_no) -> 1.0 or 0.0
    Much faster than replaying from scratch per move in build_one_compact.
    """
    capture_map = {}
    sorted_df = moves_df.sort_values(['game_id', 'move_no'])
    
    current_game_id = None
    board = None
    prev_was_capture = 0.0
    
    for rec in sorted_df[['game_id', 'move_no', 'move']].to_dict('records'):
        gid = rec['game_id']
        mno = rec['move_no']
        uci = rec.get('move', '')
        
        if gid != current_game_id:
            current_game_id = gid
            board = chess.Board()
            prev_was_capture = 0.0
        
        # Store whether the PREVIOUS move was a capture (for this position)
        capture_map[(gid, mno)] = prev_was_capture
        
        # Now make this move and track if IT is a capture (for the next position)
        prev_was_capture = 0.0
        if uci and board is not None and len(uci) >= 4:
            try:
                move = board.parse_uci(uci)
                if board.is_capture(move):
                    prev_was_capture = 1.0
                board.push(move)
            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                current_game_id = None
    
    return capture_map


def process_moves_chunked(moves_df, games_dict, poss_index, capture_map, max_possible, with_phase, workers, split_name):
    """Process moves in chunks — vectorized task building, persistent pool."""
    CHUNK_SIZE = 250000
    SHARD_SIZE = 500000

    all_examples = []
    total_processed = 0

    # Convert chunk to list-of-dicts ONCE (10-100x faster than iterrows)
    # Also pre-filter: only rows whose (game_id, move_no) exists in poss_index
    moves_records = moves_df.to_dict('records')

    # Build tasks from records (pure Python loop over dicts, no pandas overhead)
    tasks = []
    for mrow in moves_records:
        gid = mrow['game_id']
        mno = int(mrow['move_no'])
        if gid not in games_dict:
            continue
        possibles = poss_index.get((gid, mno))
        if not possibles:
            continue
        # Inject pre-computed capture flag
        mrow['_prev_capture'] = capture_map.get((gid, mno), 0.0)
        tasks.append((mrow, games_dict[gid], possibles, max_possible, with_phase))

    del moves_records
    total_tasks = len(tasks)

    if not tasks:
        return

    # Single persistent pool for entire split — no spin-up/teardown per chunk
    pool = Pool(workers) if workers > 1 else None
    try:
        for chunk_start in range(0, total_tasks, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_tasks)
            chunk_tasks = tasks[chunk_start:chunk_end]

            if pool is not None:
                results = [r for r in pool.map(build_one_compact_wrapper, chunk_tasks, chunksize=256) if r is not None]
            else:
                results = [r for r in (build_one_compact(*t) for t in chunk_tasks) if r is not None]

            all_examples.extend(results)
            total_processed += len(chunk_tasks)

            print(f"\r  {split_name}: {total_tasks:,} tasks → {total_processed:,} processed", end='', flush=True)

            while len(all_examples) >= SHARD_SIZE:
                yield all_examples[:SHARD_SIZE]
                all_examples = all_examples[SHARD_SIZE:]

            del results, chunk_tasks
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    del tasks

    if all_examples:
        yield all_examples


# ---------------------------------------------------------------------------
# Build dataset (per-worker, chunked)
# ---------------------------------------------------------------------------

def build_dataset(data_dir: str, output_dir: str, max_possible: int = 220,
                  min_elo: int = 0, max_elo: int = 0, val_frac: float = 0.1,
                  test_frac: float = 0.05, workers: int = 4, seed: int = 42,
                  with_phase: bool = False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'train').mkdir(exist_ok=True)
    (out / 'val').mkdir(exist_ok=True)
    (out / 'test').mkdir(exist_ok=True)

    root = Path(data_dir)
    print(f"[DEBUG] Data dir: {root}, exists: {root.exists()}", flush=True)
    
    # Discover worker directories at any depth under root
    worker_dirs = sorted(set(
        f.parent.parent for f in root.rglob('games/*.parquet')
    ))
    if not worker_dirs:
        print("ERROR: No worker directories found with games/*.parquet. Check data-dir path.", flush=True)
        return
    print(f"[1/4] Found {len(worker_dirs)} worker directories", flush=True)

    all_game_ids = set()
    for wdir in worker_dirs:
        games_files = sorted((wdir / 'games').glob('*.parquet'))
        for f in games_files:
            try:
                df = pq.read_table(str(f), columns=['game_id']).to_pandas()
                all_game_ids.update(df['game_id'].astype(str).tolist())
            except Exception as e:
                print(f"  Warning: could not read {f}: {e}", flush=True)
    all_game_ids = sorted(all_game_ids)
    print(f"[2/4] Found {len(all_game_ids):,} unique games", flush=True)

    if not all_game_ids:
        print("ERROR: No games found! Check data-dir path.", flush=True)
        return

    rng = np.random.RandomState(seed)
    rng.shuffle(all_game_ids)
    n_val = max(1, int(len(all_game_ids) * val_frac))
    n_test = max(1, int(len(all_game_ids) * test_frac))
    test_games = set(all_game_ids[:n_test])
    val_games = set(all_game_ids[n_test:n_test + n_val])
    train_games = set(all_game_ids[n_test + n_val:])

    split_counts = {'train': 0, 'val': 0, 'test': 0}
    shard_counters = {'train': 0, 'val': 0, 'test': 0}

    for widx, wdir in enumerate(worker_dirs):
        print(f"\n[3/4] Processing worker {widx+1}/{len(worker_dirs)}: {wdir.name}", flush=True)
        
        moves_files = sorted((wdir / 'moves').glob('*.parquet'))
        games_files = sorted((wdir / 'games').glob('*.parquet'))
        poss_files = sorted((wdir / 'possible_moves').glob('*.parquet'))
        
        if not moves_files or not games_files or not poss_files:
            print(f"  Skipping — missing files", flush=True)
            continue

        PARQUET_BATCH = 50
        # Remove duplicate definition below
        print(f"  Loading games...", flush=True)
        GAME_COLS = ['game_id', 'result', 'white_elo', 'black_elo', 'white_title', 'black_title', 'time_control']
        try:
            games_df = pd.concat([pq.read_table(str(f), columns=GAME_COLS).to_pandas() for f in games_files])
            games_df = games_df.set_index('game_id')
            if min_elo > 0 or max_elo > 0:
                mask = pd.Series(True, index=games_df.index)
                if min_elo > 0:
                    mask &= (games_df['white_elo'] >= min_elo) & (games_df['black_elo'] >= min_elo)
                if max_elo > 0:
                    mask &= (games_df['white_elo'] <= max_elo) & (games_df['black_elo'] <= max_elo)
                games_df = games_df[mask]
            games_dict = {}
            games_records = games_df.reset_index().to_dict('records')
            for rec in games_records:
                games_dict[str(rec['game_id'])] = rec
            del games_records
        except Exception as e:
            print(f"  Error loading games: {e}", flush=True)
            continue

        MOVE_COLS = ['game_id', 'move_no', 'color', 'move', 'fen_before', 'eval_before',
                     'white_win_perc_before', 'draw_perc_before', 'black_win_perc_before',
                     'time_spent', 'time_remaining', 'game_to_position']
        print(f"  Loading moves ({len(moves_files)} files)...", flush=True)
        moves_dfs = []
        for i in range(0, len(moves_files), PARQUET_BATCH):
            batch = moves_files[i:i + PARQUET_BATCH]
            batch_df = pd.concat([pq.read_table(str(f), columns=MOVE_COLS).to_pandas() for f in batch])
            batch_df['game_id'] = batch_df['game_id'].astype(str)
            batch_df = batch_df[batch_df['game_id'].isin(games_dict.keys())]
            moves_dfs.append(batch_df)
            del batch_df
        moves_df = pd.concat(moves_dfs)
        del moves_dfs
        gc.collect()
        print(f"  Loaded {len(moves_df):,} moves", flush=True)
        
        print(f"  Loading possible_moves + building index ({len(poss_files)} files)...", flush=True)
        POSS_COLS = ['game_id', 'move_no', 'move', 'eval', 'fen_after', 'to_square', 'piece',
                     'white_win_perc', 'draw_perc', 'black_win_perc', 'nodes', 'depth']
        poss_index = defaultdict(list)
        total_poss_rows = 0
        for i in range(0, len(poss_files), PARQUET_BATCH):
            batch = poss_files[i:i + PARQUET_BATCH]
            batch_df = pd.concat([pq.read_table(str(f), columns=POSS_COLS).to_pandas() for f in batch])
            batch_df['game_id'] = batch_df['game_id'].astype(str)
            batch_df = batch_df[batch_df['game_id'].isin(games_dict.keys())]
            batch_df['move_no'] = batch_df['move_no'].astype(int)
            for rec in batch_df.to_dict('records'):
                poss_index[(rec['game_id'], rec['move_no'])].append(rec)
            total_poss_rows += len(batch_df)
            print(f"    possible_moves batch {i // PARQUET_BATCH + 1}/"
                  f"{math.ceil(len(poss_files) / PARQUET_BATCH)} "
                  f"({len(batch_df):,} rows)", flush=True)
            del batch_df
        gc.collect()
        print(f"  Indexed {total_poss_rows:,} possible moves → {len(poss_index):,} positions", flush=True)

        # Split moves by game — vectorized filter instead of groupby iteration
        print(f"  Splitting moves by game...", flush=True)
        move_gids = moves_df['game_id']
        split_moves_dfs = {
            'train': moves_df[move_gids.isin(train_games)],
            'val': moves_df[move_gids.isin(val_games)],
            'test': moves_df[move_gids.isin(test_games)],
        }
        
        # Pre-compute capture flags: replay each game once instead of per-move
        print(f"  Pre-computing captures...", flush=True)
        capture_map = precompute_prev_captures(moves_df)
        print(f"  Computed {len(capture_map):,} capture flags", flush=True)
        
        for split_name in ['train', 'val', 'test']:
            split_moves_df = split_moves_dfs[split_name]
            if len(split_moves_df) == 0:
                continue
            print(f"  {split_name}: {len(split_moves_df):,} moves", end='', flush=True)
            
            # Process in chunks and write shards
            for examples in process_moves_chunked(split_moves_df, games_dict, poss_index, capture_map, max_possible, with_phase, workers, split_name):
                if not examples:
                    continue
                
                out_dir = out / split_name
                shard_id = shard_counters[split_name]
                shard_path = out_dir / f'shard_{shard_id:04d}.npz'
                
                save_dict = {
                    'fen_before': np.array([e['fen_before'] for e in examples], dtype=object),
                    'game_to_position': np.array([e['game_to_position'] for e in examples], dtype=object),
                    'possible_uci': np.array([e['possible_uci'] for e in examples], dtype=object),
                    'possible_fen_after': np.array([e['possible_fen_after'] for e in examples], dtype=object),
                    'possible_scalars': np.stack([e['possible_scalars'] for e in examples]),
                    'possible_mask': np.stack([e['possible_mask'] for e in examples]),
                    'tabular': np.stack([e['tabular'] for e in examples]),
                    'actual_idx': np.array([e['actual_idx'] for e in examples]),
                    'is_mistake': np.array([e['is_mistake'] for e in examples]),
                    'win_prob_before': np.stack([e['win_prob_before'] for e in examples]),
                    'time_spent_log': np.array([e['time_spent_log'] for e in examples]),
                }
                save_dict['game_phase'] = np.array([e.get('game_phase', 0) for e in examples])
                
                np.savez_compressed(shard_path, **save_dict)
                sz_mb = os.path.getsize(shard_path) / (1024 * 1024)
                print(f" → {len(examples):,} examples, {sz_mb:.1f} MB → {shard_path.name}", flush=True)
                
                split_counts[split_name] += len(examples)
                shard_counters[split_name] += 1
                
                del examples
                gc.collect()
        
        del games_df, moves_df, poss_index, split_moves_dfs, capture_map
        gc.collect()

    print(f"\n[4/4] Total examples: train={split_counts['train']:,}, val={split_counts['val']:,}, test={split_counts['test']:,}", flush=True)
    
    config = {
        'format': 'compact_sharded_v1',
        'max_possible': max_possible,
        'min_elo': min_elo,
        'max_elo': max_elo,
        'with_phase': with_phase,
        'val_frac': val_frac,
        'test_frac': test_frac,
        'seed': seed,
    }
    with open(out / 'dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Done!", flush=True)


if __name__ == '__main__':
    print("Starting mimo_dataset.py...", flush=True)
    parser = argparse.ArgumentParser(description='Build MIMO chess dataset (compact, per-worker, chunked)')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-possible', type=int, default=220)
    parser.add_argument('--min-elo', type=int, default=0)
    parser.add_argument('--max-elo', type=int, default=0)
    parser.add_argument('--val-frac', type=float, default=0.10)
    parser.add_argument('--test-frac', type=float, default=0.05)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--with-phase', action='store_true', default=True,
                        help='Compute game phase labels (default: True)')
    parser.add_argument('--no-phase', dest='with_phase', action='store_false',
                        help='Skip game phase computation')
    args = parser.parse_args()
    print(f"Args: data_dir={args.data_dir}, output_dir={args.output_dir}, workers={args.workers}", flush=True)
    build_dataset(args.data_dir, args.output_dir, args.max_possible,
                  args.min_elo, args.max_elo, args.val_frac, args.test_frac,
                  args.workers, args.seed, args.with_phase)
