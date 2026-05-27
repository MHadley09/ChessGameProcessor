#!/usr/bin/env python3
"""
mimo_dataset.py — Compact dataset for the MIMO chess model (per-worker streaming, chunked processing).

Reads from Parquet files produced by LC0 processing pipeline.
Processes one worker directory at a time with chunked task generation to avoid OOM.
Stores compact data: FENs, UCI moves, scalars, labels. Planes built on-the-fly.

This version processes moves in chunks of 100K to avoid building giant task lists.
"""

import argparse
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
import duckdb
from torch.utils.data import Dataset

print("Imports complete, starting...", flush=True)


# ---------------------------------------------------------------------------
# Plane codec
# ---------------------------------------------------------------------------

def board_to_planes(board: chess.Board, history: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
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
                for pt in range(1, 7):
                    for color in (chess.WHITE, chess.BLACK):
                        idx = 12 + (pt - 1) + 6 * color
                        for sq in temp.pieces(pt, color):
                            planes[idx, 7 - sq // 8, sq % 8] = 1.0
        except Exception:
            pass
        try:
            temp2 = board.copy()
            if len(temp2.move_stack) >= 2 and len(history) >= 2:
                temp2.pop()
                temp2.pop()
                for pt in range(1, 7):
                    for color in (chess.WHITE, chess.BLACK):
                        idx = 24 + (pt - 1) + 6 * color
                        for sq in temp2.pieces(pt, color):
                            planes[idx, 7 - sq // 8, sq % 8] = 1.0
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


def result_to_wdl(result: str, color: str) -> np.ndarray:
    if result == '1-0':
        return np.array([1., 0., 0.], dtype=np.float32) if color == 'White' else np.array([0., 0., 1.], dtype=np.float32)
    elif result == '0-1':
        return np.array([1., 0., 0.], dtype=np.float32) if color == 'Black' else np.array([0., 0., 1.], dtype=np.float32)
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
    moves = parse_game_to_position(game_to_position_str)
    if not moves:
        return 0.0
    try:
        replay = chess.Board()
        was_capture = False
        for from_sq, to_sq in moves:
            for legal in replay.legal_moves:
                if legal.from_square == from_sq and legal.to_square == to_sq:
                    was_capture = replay.is_capture(legal)
                    replay.push(legal)
                    break
        return 1.0 if was_capture else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Build one example (top-level for pickling)
# ---------------------------------------------------------------------------

def build_one_compact(move_dict: Dict, game_dict: Dict, possibles: List[Dict], max_possible: int, with_phase: bool) -> Optional[Dict[str, Any]]:
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

    evals_stm = []
    for pm in possibles:
        e = _safe_float(pm.get('eval'), 0)
        evals_stm.append(e if color == 'White' else -e)
    best_eval_stm = max(evals_stm) if evals_stm else 0.0
    worst_eval_stm = min(evals_stm) if evals_stm else 0.0
    eval_range = best_eval_stm - worst_eval_stm

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
            _safe_float(pm.get('policy_prob'), 0.0),
        ], dtype=np.float32))

    while len(poss_scalars_list) < max_possible:
        poss_scalars_list.append(np.zeros(12, dtype=np.float32))
        poss_uci_list.append('')
        poss_fen_after_list.append('')

    possible_mask = np.zeros(max_possible, dtype=np.float32)
    possible_mask[:num_possible] = 1.0

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

    actual_uci = move_dict.get('move', '')
    actual_idx = -1
    for i, pm in enumerate(possibles):
        if pm.get('move') == actual_uci:
            actual_idx = i
            break

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

    wdl_before = result_to_wdl(game_result, color)
    color_after = 'Black' if color == 'White' else 'White'
    wdl_after = result_to_wdl(game_result, color_after)
    raw_ts = max(0.0, _safe_float(move_dict.get('time_spent')))
    time_spent_log = np.float32(math.log1p(raw_ts))
    gtp = str(move_dict.get('game_to_position', '')) if move_dict.get('game_to_position') else ''

    result = {
        'fen_before': fen_before,
        'game_to_position': gtp,
        'possible_uci': poss_uci_list,
        'possible_fen_after': poss_fen_after_list,
        'possible_scalars': np.stack(poss_scalars_list),
        'possible_mask': possible_mask,
        'tabular': tabular,
        'actual_idx': np.int64(actual_idx),
        'is_mistake': np.float32(is_mistake),
        'win_prob_before': wdl_before,
        'win_prob_after': wdl_after,
        'time_spent_log': time_spent_log,
    }
    
    if with_phase:
        game_phase = 0
        try:
            ply = int(_safe_float(move_dict.get('move_no'), 0))
            if ply <= 20:
                game_phase = 0
            elif ply <= 60:
                game_phase = 1
            else:
                game_phase = 2
        except:
            pass
        result['game_phase'] = np.int64(game_phase)
    
    return result


# Wrapper for executor (picklable)
def build_one_compact_wrapper(args_tuple):
    return build_one_compact(*args_tuple)


# ---------------------------------------------------------------------------
# Sharded Dataset
# ---------------------------------------------------------------------------

class MIMOCompactDataset(Dataset):
    def __init__(self, data_path: str, max_possible: int = 40, cache_shards: int = 3, with_phase: bool = False):
        self.data_path = Path(data_path)
        self.max_possible = max_possible
        self.cache_shards = cache_shards
        self.with_phase = with_phase
        
        if self.data_path.is_dir():
            self.shard_files = sorted([f for f in self.data_path.glob('*.npz')])
            if not self.shard_files:
                raise FileNotFoundError(f"No .npz shards in {data_path}")
            self.shard_offsets = []
            self.shard_counts = []
            total = 0
            for f in self.shard_files:
                with np.load(f, allow_pickle=True) as data:
                    n = len(data['fen_before'])
                    self.shard_offsets.append(total)
                    self.shard_counts.append(n)
                    total += n
            self.n = total
            self._shard_cache = OrderedDict()
        else:
            data = np.load(self.data_path, allow_pickle=True)
            self.fen_before = data['fen_before']
            self.game_to_position = data['game_to_position']
            self.possible_uci = data['possible_uci']
            self.possible_fen_after = data['possible_fen_after']
            self.possible_scalars = data['possible_scalars']
            self.possible_mask = data['possible_mask']
            self.tabular = data['tabular']
            self.actual_idx = data['actual_idx']
            self.is_mistake = data['is_mistake']
            self.win_prob_before = data['win_prob_before']
            self.win_prob_after = data['win_prob_after']
            self.time_spent_log = data['time_spent_log']
            if with_phase and 'game_phase' in data:
                self.game_phase = data['game_phase']
            self.n = len(self.fen_before)
            self.shard_files = None
        print(f"[DATA] {self.n:,} examples", flush=True)

    def __len__(self):
        return self.n

    def _load_shard(self, shard_idx):
        if shard_idx in self._shard_cache:
            self._shard_cache.move_to_end(shard_idx)
            return self._shard_cache[shard_idx]
        while len(self._shard_cache) >= self.cache_shards:
            self._shard_cache.popitem(last=False)
        data = np.load(self.shard_files[shard_idx], allow_pickle=True)
        shard_data = {k: data[k] for k in data.files}
        self._shard_cache[shard_idx] = shard_data
        return shard_data

    def __getitem__(self, idx):
        if self.shard_files:
            shard_idx = 0
            while shard_idx < len(self.shard_offsets) - 1 and idx >= self.shard_offsets[shard_idx + 1]:
                shard_idx += 1
            local_idx = idx - self.shard_offsets[shard_idx]
            data = self._load_shard(shard_idx)
            fen_before = str(data['fen_before'][local_idx])
            gtp = str(data['game_to_position'][local_idx])
            possible_uci = data['possible_uci'][local_idx]
            possible_fen_after = data['possible_fen_after'][local_idx]
            possible_scalars = data['possible_scalars'][local_idx]
            possible_mask = data['possible_mask'][local_idx]
            tabular = data['tabular'][local_idx]
            actual_idx = int(data['actual_idx'][local_idx])
            is_mistake = float(data['is_mistake'][local_idx])
            win_prob_before = data['win_prob_before'][local_idx]
            win_prob_after = data['win_prob_after'][local_idx]
            time_spent_log = float(data['time_spent_log'][local_idx])
            game_phase = int(data.get('game_phase', [0]*len(data['fen_before']))[local_idx]) if self.with_phase else 0
        else:
            fen_before = str(self.fen_before[idx])
            gtp = str(self.game_to_position[idx])
            possible_uci = self.possible_uci[idx]
            possible_fen_after = self.possible_fen_after[idx]
            possible_scalars = self.possible_scalars[idx]
            possible_mask = self.possible_mask[idx]
            tabular = self.tabular[idx]
            actual_idx = int(self.actual_idx[idx])
            is_mistake = float(self.is_mistake[idx])
            win_prob_before = self.win_prob_before[idx]
            win_prob_after = self.win_prob_after[idx]
            time_spent_log = float(self.time_spent_log[idx])
            game_phase = int(self.game_phase[idx]) if self.with_phase and hasattr(self, 'game_phase') else 0

        try:
            board = chess.Board(fen_before)
            history = parse_game_to_position(gtp)
            recent = history[-2:] if history else []
            current_planes = board_to_planes(board, recent)
        except Exception:
            current_planes = np.zeros((47, 8, 8), dtype=np.float32)
            recent = []

        poss_planes = np.zeros((self.max_possible, 47, 8, 8), dtype=np.float32)
        for i in range(self.max_possible):
            if possible_mask[i] < 0.5:
                break
            fen_after = str(possible_fen_after[i]) if i < len(possible_fen_after) else ''
            uci = str(possible_uci[i]) if i < len(possible_uci) else ''
            if not fen_after or not uci:
                continue
            try:
                board_after = chess.Board(fen_after)
                from_sq = chess.parse_square(uci[:2])
                to_sq = chess.parse_square(uci[2:4])
                poss_hist = [(from_sq, to_sq)] + (recent[:1] if recent else [])
                poss_planes[i] = board_to_planes(board_after, poss_hist)
            except Exception:
                pass

        out = {
            'current_planes': torch.from_numpy(current_planes).float(),
            'possible_planes': torch.from_numpy(poss_planes).float(),
            'possible_scalars': torch.from_numpy(possible_scalars).float(),
            'possible_mask': torch.from_numpy(possible_mask).float(),
            'tabular': torch.from_numpy(tabular).float(),
            'actual_idx': torch.tensor(actual_idx, dtype=torch.long),
            'is_mistake': torch.tensor(is_mistake, dtype=torch.float32),
            'win_prob_before': torch.from_numpy(win_prob_before).float(),
            'win_prob_after': torch.from_numpy(win_prob_after).float(),
            'time_spent_log': torch.tensor(time_spent_log, dtype=torch.float32),
        }
        if self.with_phase:
            out['game_phase'] = torch.tensor(game_phase, dtype=torch.long)
        return out


# ---------------------------------------------------------------------------
# Process moves in chunks
# ---------------------------------------------------------------------------

def process_moves_chunked(moves_df, games_dict, con, max_possible, with_phase, workers, split_name):
    """Process moves in chunks using DuckDB for possibles lookup."""
    CHUNK_SIZE = 100000
    SHARD_SIZE = 500000
    all_examples = []
    total_processed = 0
    for chunk_start in range(0, len(moves_df), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(moves_df))
        chunk_df = moves_df.iloc[chunk_start:chunk_end]
        # Query possibles for this chunk
        pairs_df = chunk_df[['game_id','move_no']].drop_duplicates()
        con.register('pairs', pairs_df)
        poss_chunk = con.execute('SELECT p.* FROM poss p JOIN pairs USING (game_id, move_no)').fetchdf()
        # Build temporary index for this chunk
        poss_index_chunk = {}
        if not poss_chunk.empty:
            for (gid, mno), group in poss_chunk.groupby(['game_id','move_no']):
                poss_index_chunk[(str(gid), int(mno))] = group.to_dict('records')
        tasks = []
        for _, mrow in chunk_df.iterrows():
            gid = mrow['game_id']
            mno = int(mrow['move_no'])
            if gid not in games_dict:
                continue
            possibles = poss_index_chunk.get((gid, mno), [])
            if not possibles:
                continue
            tasks.append((mrow.to_dict(), games_dict[gid], possibles, max_possible, with_phase))
        if not tasks:
            continue
        results = []
        if workers > 1 and tasks:
            with Pool(workers) as pool:
                results = [r for r in pool.map(build_one_compact_wrapper, tasks, chunksize=64) if r is not None]
        else:
            results = [build_one_compact(*t) for t in tasks]
            results = [r for r in results if r is not None]
        all_examples.extend(results)
        total_processed += len(chunk_df)
        print(f"  {split_name}: {len(moves_df):,} moves → {total_processed:,} processed", end='', flush=True)
        if len(all_examples) >= SHARD_SIZE:
            yield all_examples[:SHARD_SIZE]
            all_examples = all_examples[SHARD_SIZE:]
        del tasks, results, chunk_df, pairs_df, poss_chunk, poss_index_chunk
        gc.collect()
    if all_examples:
        yield all_examples


# ---------------------------------------------------------------------------
# Build dataset (per-worker, chunked)
# ---------------------------------------------------------------------------

def build_dataset(data_dir: str, output_dir: str, max_possible: int = 40,
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
        try:
            games_df = pd.concat([pq.read_table(str(f)).to_pandas() for f in games_files])
            games_df = games_df.set_index('game_id')
            if min_elo > 0 or max_elo > 0:
                mask = pd.Series(True, index=games_df.index)
                if min_elo > 0:
                    mask &= (games_df['white_elo'] >= min_elo) & (games_df['black_elo'] >= min_elo)
                if max_elo > 0:
                    mask &= (games_df['white_elo'] <= max_elo) & (games_df['black_elo'] <= max_elo)
                games_df = games_df[mask]
            games_dict = {str(k): v.to_dict() for k, v in games_df.iterrows()}
        except Exception as e:
            print(f"  Error loading games: {e}", flush=True)
            continue

        print(f"  Loading moves ({len(moves_files)} files)...", flush=True)
        moves_dfs = []
        for i in range(0, len(moves_files), PARQUET_BATCH):
            batch = moves_files[i:i + PARQUET_BATCH]
            batch_df = pd.concat([pq.read_table(str(f)).to_pandas() for f in batch])
            batch_df['game_id'] = batch_df['game_id'].astype(str)
            batch_df = batch_df[batch_df['game_id'].isin(games_dict.keys())]
            moves_dfs.append(batch_df)
            del batch_df
        moves_df = pd.concat(moves_dfs)
        del moves_dfs
        gc.collect()
        print(f"  Loaded {len(moves_df):,} moves", flush=True)
        
        print(f"  Loading possible_moves ({len(poss_files)} files)...", flush=True)
        poss_dfs = []
        for i in range(0, len(poss_files), PARQUET_BATCH):
            batch = poss_files[i:i + PARQUET_BATCH]
            batch_df = pd.concat([pq.read_table(str(f)).to_pandas() for f in batch])
            batch_df['game_id'] = batch_df['game_id'].astype(str)
            batch_df = batch_df[batch_df['game_id'].isin(games_dict.keys())]
            poss_dfs.append(batch_df)
            print(f"    possible_moves batch {i // PARQUET_BATCH + 1}/"
                  f"{math.ceil(len(poss_files) / PARQUET_BATCH)} "
                  f"({len(batch_df):,} rows)", flush=True)
            del batch_df
        poss_df = pd.concat(poss_dfs)
        del poss_dfs
        gc.collect()
        print(f"  Loaded {len(poss_df):,} possible moves", flush=True)
        
        # Register possible_moves in DuckDB
        print(f"  Registering possible_moves in DuckDB...", flush=True)
        con = duckdb.connect(database=':memory:')
        poss_df['game_id'] = poss_df['game_id'].astype(str)
        poss_df['move_no'] = poss_df['move_no'].astype(int)
        con.register('poss', poss_df)
        print(f"  Registered {len(poss_df):,} possible moves", flush=True)
        # poss_df kept registered; will be garbage collected with con


        # Split moves by game
        print(f"  Splitting moves by game...", flush=True)
        split_moves_dfs = {'train': [], 'val': [], 'test': []}
        for gid, group in moves_df.groupby('game_id'):
            if gid in train_games:
                split_moves_dfs['train'].append(group)
            elif gid in val_games:
                split_moves_dfs['val'].append(group)
            elif gid in test_games:
                split_moves_dfs['test'].append(group)
        
        for split_name in ['train', 'val', 'test']:
            dfs = split_moves_dfs[split_name]
            if not dfs:
                continue
            split_moves_df = pd.concat(dfs)
            print(f"  {split_name}: {len(split_moves_df):,} moves", end='', flush=True)
            
            # Process in chunks and write shards
            for examples in process_moves_chunked(split_moves_df, games_dict, con, max_possible, with_phase, workers, split_name):
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
                    'win_prob_after': np.stack([e['win_prob_after'] for e in examples]),
                    'time_spent_log': np.array([e['time_spent_log'] for e in examples]),
                }
                if with_phase:
                    save_dict['game_phase'] = np.array([e.get('game_phase', 0) for e in examples])
                
                np.savez_compressed(shard_path, **save_dict)
                sz_mb = os.path.getsize(shard_path) / (1024 * 1024)
                print(f" → {len(examples):,} examples, {sz_mb:.1f} MB → {shard_path.name}", flush=True)
                
                split_counts[split_name] += len(examples)
                shard_counters[split_name] += 1
                
                del examples
                gc.collect()
        
        del games_df, moves_df, poss_index, split_moves_dfs
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
    parser.add_argument('--max-possible', type=int, default=40)
    parser.add_argument('--min-elo', type=int, default=0)
    parser.add_argument('--max-elo', type=int, default=0)
    parser.add_argument('--val-frac', type=float, default=0.10)
    parser.add_argument('--test-frac', type=float, default=0.05)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--with-phase', action='store_true')
    args = parser.parse_args()
    print(f"Args: data_dir={args.data_dir}, output_dir={args.output_dir}, workers={args.workers}", flush=True)
    build_dataset(args.data_dir, args.output_dir, args.max_possible,
                  args.min_elo, args.max_elo, args.val_frac, args.test_frac,
                  args.workers, args.seed, args.with_phase)
