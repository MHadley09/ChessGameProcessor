"""
DatasetWriter — Builds MIMO dataset examples on-the-fly and writes them
directly to compressed .npz shard files compatible with MIMOCompactDataset.

Each worker maintains its own DatasetWriter instance. When a game is
fully processed, the writer receives the game record, all move records,
and all possible_move records at once. It then:
  1. Groups possible moves by move_no (trivial — they arrive per-game)
  2. Builds dataset examples using build_one_compact logic
  3. Buffers examples and flushes to NPZ shards at configurable intervals

Output structure:
    output_dir/train/shard_wXX_YYYYMMDD_HHMMSS_XXXX.npz
    output_dir/val/shard_wXX_YYYYMMDD_HHMMSS_XXXX.npz
    output_dir/test/shard_wXX_YYYYMMDD_HHMMSS_XXXX.npz

Shard format matches dataset_v4.py exactly:
    fen_before, game_to_position, possible_uci, possible_fen_after,
    possible_scalars, possible_mask, tabular, actual_idx, is_mistake,
    win_prob_before, time_spent_log
"""

import gc
import glob
import math
import os
import re
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import numpy as np


# ---------------------------------------------------------------------------
# Pre-allocated constants
# ---------------------------------------------------------------------------

_WDL_WHITE_WIN = np.array([1., 0., 0.], dtype=np.float32)
_WDL_DRAW      = np.array([0., 1., 0.], dtype=np.float32)
_WDL_BLACK_WIN = np.array([0., 0., 1.], dtype=np.float32)


def result_to_wdl(result: str) -> np.ndarray:
    """Game outcome from White's perspective."""
    if result == '1-0':
        return _WDL_WHITE_WIN.copy()
    elif result == '0-1':
        return _WDL_BLACK_WIN.copy()
    else:
        return _WDL_DRAW.copy()


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        v = float(val)
        if v != v:  # NaN check
            return default
        return v
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _compute_game_phase(board: chess.Board, move_dict: Dict) -> int:
    """
    Compute game phase: 0=opening, 1=middlegame, 2=endgame.
    
    Opening: ply <= 14 (move 7) AND not in endgame material.
    Endgame: material-based — no queens on board, OR total non-pawn
             material (knights, bishops, rooks) <= 3 pieces combined.
    Middlegame: everything else.
    """
    ply = int(_safe_float(move_dict.get('move_no'), 0))

    # Count material
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    minors_majors = (len(board.pieces(chess.KNIGHT, chess.WHITE)) +
                     len(board.pieces(chess.BISHOP, chess.WHITE)) +
                     len(board.pieces(chess.ROOK, chess.WHITE)) +
                     len(board.pieces(chess.KNIGHT, chess.BLACK)) +
                     len(board.pieces(chess.BISHOP, chess.BLACK)) +
                     len(board.pieces(chess.ROOK, chess.BLACK)))

    # Endgame: no queens, or very low material
    if queens == 0 or (queens == 0 and minors_majors <= 3):
        return 2

    # Opening: ply <= 14 (move 7)
    if ply <= 14:
        return 0

    # Middlegame
    return 1


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


# ---------------------------------------------------------------------------
# Build one MIMO example — same logic as dataset_v4.py build_one_compact
# ---------------------------------------------------------------------------

def build_example(
    move_dict: Dict,
    game_dict: Dict,
    possibles: List[Dict],
    max_possible: int,
    prev_was_capture: float,
    with_phase: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Build a single MIMO dataset example from a move + its possible moves.

    Args:
        move_dict: Actual move record from _process_game_with_engine
        game_dict: Game-level metadata
        possibles: List of possible_move dicts for this position
        max_possible: Max candidate moves to keep (220)
        prev_was_capture: 1.0 if the previous move was a capture, else 0.0
        with_phase: Whether to include game_phase label

    Returns:
        Dict with all shard fields, or None if example should be skipped
    """
    fen_before = move_dict.get('fen_before', '')
    try:
        board = chess.Board(fen_before)
    except Exception:
        return None
    if not possibles:
        return None

    color = move_dict.get('color', 'white')
    color_norm = color.lower()

    game_result = game_dict.get('result', '1/2-1/2')
    w_elo = _safe_float(game_dict.get('white_elo'), 1500)
    b_elo = _safe_float(game_dict.get('black_elo'), 1500)

    # Sort by eval descending, keep top max_possible
    possibles = sorted(possibles, key=lambda x: _safe_float(x.get('eval'), -99999), reverse=True)
    possibles = possibles[:max_possible]
    num_possible = len(possibles)

    is_white = color_norm == 'white'
    sign = 1 if is_white else -1

    evals_stm = []
    for pm in possibles:
        evals_stm.append(_safe_float(pm.get('eval'), 0) * sign)
    best_eval_stm = max(evals_stm) if evals_stm else 0.0
    worst_eval_stm = min(evals_stm) if evals_stm else 0.0
    eval_range = best_eval_stm - worst_eval_stm
    inv_eval_range = 1.0 / eval_range if eval_range > 0 else 0.0

    # Pre-compute check/checkmate by push/pop on single board
    legal_move_info = {}
    for move in board.legal_moves:
        uci = move.uci()
        board.push(move)
        legal_move_info[uci] = (board.is_check(), board.is_checkmate())
        board.pop()

    poss_scalars = np.zeros((max_possible, 12), dtype=np.float32)
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
            to_sq_str = pm.get('to_square', '')
            to_sq_int = chess.parse_square(to_sq_str)
            is_capture = 1.0 if board.piece_at(to_sq_int) is not None else 0.0
            if board.ep_square == to_sq_int and str(pm.get('piece', '')).upper() == 'P':
                is_capture = 1.0
        except Exception:
            is_capture = 0.0

        # Check/checkmate from pre-computed dict
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
            0.0,  # policy_prob — unused, always 0
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
    in_check = 1.0 if board.is_check() else 0.0
    eval_std = float(np.std(evals_stm)) / 1000.0 if len(evals_stm) > 1 else 0.0

    # Aggregate from pre-built scalars
    if num_possible > 0:
        num_captures = float(poss_scalars[:num_possible, 8].sum()) / num_possible
        num_checks = float(poss_scalars[:num_possible, 9].sum()) / num_possible
    else:
        num_captures = 0.0
        num_checks = 0.0
    num_candidates = num_possible / max_possible

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
        prev_was_capture,
        in_check,
        eval_std,
        num_captures,
        num_checks,
        num_candidates,
    ], dtype=np.float32)

    # Mistake detection + per-move classification
    is_mistake = 0.0
    frac_mistake_moves = 0.0
    frac_excellent_moves = max(1.0 / num_possible, 0.0) if num_possible > 0 else 0.0  # minimum 1

    if actual_idx >= 0 and num_possible > 0:
        def _expected_score_stm(pm_dict):
            """Expected score from side-to-move perspective (0-100 scale)."""
            w = _safe_float(pm_dict.get('white_win_perc'), 33.0)
            d = _safe_float(pm_dict.get('draw_perc'), 34.0)
            white_es = w + 0.5 * d
            return white_es if is_white else (100.0 - white_es)

        best_idx = max(range(num_possible), key=lambda i: _expected_score_stm(possibles[i]))
        best_es = _expected_score_stm(possibles[best_idx])
        played_es = _expected_score_stm(possibles[actual_idx])
        drop = best_es - played_es
        avg_elo = (w_elo + b_elo) / 2
        # Thresholds on 0-100 scale to match WDL percentages
        threshold = 20.0 if avg_elo < 1500 else (15.0 if avg_elo < 2500 else 10.0)
        if drop > threshold:
            is_mistake = 1.0

        if drop > 5.0 and is_mistake == 0.0:
            def _outcome_class_stm(pm_dict):
                """Outcome class from STM perspective: 2=winning, 1=drawing, 0=losing."""
                w = _safe_float(pm_dict.get('white_win_perc'), 33.0)
                d = _safe_float(pm_dict.get('draw_perc'), 34.0)
                l = _safe_float(pm_dict.get('black_win_perc'), 33.0)
                if is_white:
                    mx = max(w, d, l)
                    if mx == w: return 2
                    elif mx == d: return 1
                    return 0
                else:
                    mx = max(w, d, l)
                    if mx == l: return 2   # Black winning
                    elif mx == d: return 1
                    return 0
            if _outcome_class_stm(possibles[actual_idx]) < _outcome_class_stm(possibles[best_idx]):
                is_mistake = 1.0

        # Classify all legal moves for frac features
        n_excellent = 0
        n_mistake = 0
        for j in range(num_possible):
            j_es = _expected_score_stm(possibles[j])
            j_drop = best_es - j_es
            j_mistake = False
            if j_drop > threshold:
                j_mistake = True
            elif j_drop > 5.0:
                def _oc_stm(pm_dict):
                    w = _safe_float(pm_dict.get('white_win_perc'), 33.0)
                    d = _safe_float(pm_dict.get('draw_perc'), 34.0)
                    l = _safe_float(pm_dict.get('black_win_perc'), 33.0)
                    if is_white:
                        mx = max(w, d, l)
                        if mx == w: return 2
                        elif mx == d: return 1
                        return 0
                    else:
                        mx = max(w, d, l)
                        if mx == l: return 2
                        elif mx == d: return 1
                        return 0
                if _oc_stm(possibles[j]) < _oc_stm(possibles[best_idx]):
                    j_mistake = True
            if j_mistake:
                n_mistake += 1
            elif j_drop <= 2.0:
                n_excellent += 1

        frac_mistake_moves = n_mistake / num_possible
        frac_excellent_moves = max(n_excellent, 1) / num_possible

    # Append frac_mistake_moves and frac_excellent_moves to tabular (slots 18, 19)
    tabular = np.append(tabular, [frac_mistake_moves, frac_excellent_moves]).astype(np.float32)

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

    if with_phase:
        game_phase = _compute_game_phase(board, move_dict)
        result['game_phase'] = np.int64(game_phase)

    return result


# ---------------------------------------------------------------------------
# DatasetWriter — buffered NPZ shard writer
# ---------------------------------------------------------------------------

class DatasetWriter:
    """
    Buffered NPZ shard writer. Receives complete game data and immediately
    builds dataset examples, buffering them until a shard is full.

    Thread-safe: uses a lock around buffer mutations for the result
    collector thread pattern.

    Train/val/test split is deterministic based on game_hash:
        hash_int % 100 < test_pct  → test
        hash_int % 100 < test_pct + val_pct  → val
        else → train
    """

    def __init__(
        self,
        output_dir: str,
        worker_id: int = 0,
        max_possible: int = 220,
        shard_size: int = 5_000,
        val_pct: int = 10,
        test_pct: int = 10,
        with_phase: bool = False,
        run_timestamp: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.worker_id = worker_id
        self.max_possible = max_possible
        self.shard_size = shard_size
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.with_phase = with_phase
        self.run_timestamp = run_timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create split directories
        for split in ('train', 'val', 'test'):
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)

        # Per-split buffers: list of game chunks (each chunk is one game's examples).
        # Flushing always cuts at game boundaries so all moves from one game
        # stay in the same shard.
        self._buffers: Dict[str, List[List[Dict]]] = {
            'train': [], 'val': [], 'test': [],
        }
        self._buffer_lens: Dict[str, int] = {
            'train': 0, 'val': 0, 'test': 0,
        }
        # Scan existing shards to avoid overwriting previous runs
        self._shard_counters: Dict[str, int] = {}
        for split in ('train', 'val', 'test'):
            self._shard_counters[split] = self._next_shard_id(split)
        self._lock = threading.Lock()

        # Counters
        self.games_written = 0
        self.moves_written = 0
        self.examples_written = 0
        self.possible_moves_written = 0

    def _next_shard_id(self, split: str) -> int:
        """Scan existing shards for this worker in the split directory
        and return the next available shard ID to prevent overwrites."""
        split_dir = self.output_dir / split
        if not split_dir.exists():
            return 0
        max_id = -1
        pattern = re.compile(
            rf'^shard_w{self.worker_id:02d}_\d{{8}}_\d{{6}}_(\d{{4}})\.npz$'
        )
        # Also match old format without timestamp for backward compat
        pattern_old = re.compile(
            rf'^shard_w{self.worker_id:02d}_(\d{{4}})\.npz$'
        )
        for f in os.listdir(split_dir):
            m = pattern.match(f) or pattern_old.match(f)
            if m:
                max_id = max(max_id, int(m.group(1)))
        return max_id + 1

    def _get_split(self, game_hash: str) -> str:
        """Deterministic train/val/test split from game hash."""
        h = int(game_hash[:8], 16) % 100
        if h < self.test_pct:
            return 'test'
        elif h < self.test_pct + self.val_pct:
            return 'val'
        return 'train'

    def write_game_data(
        self,
        game_rec: Dict,
        move_records: List[Dict],
        possible_move_records: List[Dict],
    ):
        """
        Process a complete game and buffer the resulting dataset examples.

        Has all data for one game at once, so examples are built immediately
        without any index/join step.

        Crash safety: all examples for a game are built into a temporary
        list first. Only after the entire game succeeds are they committed
        to the main shard buffer. If an exception occurs mid-game, no
        partial examples leak into any shard.

        Implements pre-computed capture tracking: replays the game once
        to track which moves were captures, passing that to each example.
        """
        game_hash = game_rec.get('game_hash', '')
        split = self._get_split(game_hash)

        # Group possible moves by move_no
        poss_by_move: Dict[int, List[Dict]] = defaultdict(list)
        for pm in possible_move_records:
            poss_by_move[int(pm.get('move_no', 0))].append(pm)

        # Pre-compute captures: replay game once, track capture per move.
        # This eliminates the quadratic replay that detect_prev_capture did.
        prev_was_capture = 0.0
        capture_by_move: Dict[int, float] = {}
        try:
            replay_board = chess.Board()
            for mr in move_records:
                move_no = int(mr.get('move_no', 0))
                capture_by_move[move_no] = prev_was_capture
                uci = mr.get('move', '')
                prev_was_capture = 0.0
                if uci and len(uci) >= 4:
                    try:
                        move_obj = replay_board.parse_uci(uci)
                        if replay_board.is_capture(move_obj):
                            prev_was_capture = 1.0
                        replay_board.push(move_obj)
                    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                        pass
        except Exception:
            pass

        # Build all examples for this game into a temporary list.
        # Nothing is committed to the shard buffer until every move
        # in the game has been processed successfully.
        game_examples = []
        for mr in move_records:
            move_no = int(mr.get('move_no', 0))
            possibles = poss_by_move.get(move_no)
            if not possibles:
                continue

            prev_cap = capture_by_move.get(move_no, 0.0)
            example = build_example(
                mr, game_rec, possibles,
                self.max_possible, prev_cap, self.with_phase,
            )
            if example is not None:
                game_examples.append(example)

        if not game_examples:
            return

        # Commit: all examples built successfully, add to shard buffer
        with self._lock:
            self._buffers[split].append(game_examples)
            self._buffer_lens[split] += len(game_examples)
            self.games_written += 1
            self.moves_written += len(move_records)
            self.examples_written += len(game_examples)
            self.possible_moves_written += len(possible_move_records)

            # Flush when buffer has enough games (shard_size = games per shard)
            for s in ('train', 'val', 'test'):
                while len(self._buffers[s]) >= self.shard_size:
                    shard_games = self._buffers[s][:self.shard_size]
                    self._buffers[s] = self._buffers[s][self.shard_size:]
                    flat = [e for game in shard_games for e in game]
                    self._buffer_lens[s] -= len(flat)
                    self._flush_shard(s, flat)

    def _flush_shard(self, split: str, examples: List[Dict]):
        """Write a list of examples to a compressed NPZ shard.

        Atomic write: data is written to a .tmp file first, then
        os.replace()'d to the final name. This prevents corrupt partial
        NPZ files if the process crashes mid-write.

        All examples from the same game are always in the same shard
        (buffer is game-chunked, flush cuts at game boundaries).
        """
        if not examples:
            return

        shard_id = self._shard_counters[split]
        shard_path = self.output_dir / split / f'shard_w{self.worker_id:02d}_{self.run_timestamp}_{shard_id:04d}.npz'
        # np.savez_compressed auto-appends .npz if missing, so use a .tmp
        # file without .npz suffix and let numpy create .tmp.npz
        tmp_stem = shard_path.parent / f'.tmp_w{self.worker_id:02d}_{self.run_timestamp}_{shard_id:04d}'
        tmp_actual = tmp_stem.parent / f'{tmp_stem.name}.npz'  # what numpy will create

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
        if self.with_phase:
            save_dict['game_phase'] = np.array([e.get('game_phase', 0) for e in examples])

        np.savez_compressed(str(tmp_stem), **save_dict)
        os.replace(str(tmp_actual), str(shard_path))

        sz_mb = os.path.getsize(shard_path) / (1024 * 1024)
        print(
            f"[W{self.worker_id:02d}] Shard {split}/{shard_path.name}: "
            f"{len(examples):,} examples, {sz_mb:.1f} MB",
            flush=True,
        )

        self._shard_counters[split] = shard_id + 1
        del examples
        gc.collect()

    def flush(self):
        """Flush all remaining buffered examples to shards."""
        with self._lock:
            for split in ('train', 'val', 'test'):
                if self._buffers[split]:
                    flat = [e for game in self._buffers[split] for e in game]
                    self._flush_shard(split, flat)
                    self._buffers[split] = []
                    self._buffer_lens[split] = 0

    def close(self):
        """Flush and finalize."""
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
