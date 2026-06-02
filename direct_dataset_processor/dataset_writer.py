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
    output_dir/train/shard_wXX_XXXX.npz
    output_dir/val/shard_wXX_XXXX.npz
    output_dir/test/shard_wXX_XXXX.npz

Shard format matches dataset_v4.py exactly:
    fen_before, game_to_position, possible_uci, possible_fen_after,
    possible_scalars (13-dim: ..., is_mistake_move, is_excellent_move),
    possible_mask, tabular (20-dim: ..., frac_mistake_moves, frac_excellent_moves),
    actual_idx, is_mistake, win_prob_before, time_spent_log

Mistake definition: EV = W*1.0 + D*0.5 + L*0.0 from side-to-move perspective.
  - Mistake: EV drop from best move > 0.25
  - Excellent: EV drop from best move < 0.025, or IS the best move
"""

import gc
import math
import os
import threading
from collections import defaultdict
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

    # ------------------------------------------------------------------
    # STM Expected Value + per-move mistake/excellent flags
    # EV = win * 1.0 + draw * 0.5 + loss * 0.0 (from side-to-move perspective)
    # WDL in poss_scalars slots 1-3 are white_win%, draw%, black_win% on 0-100
    # ------------------------------------------------------------------
    is_mistake = 0.0
    frac_mistake_moves = 0.0
    frac_excellent_moves = 0.0

    if num_possible > 0:
        # Compute STM expected value for each candidate move
        w_pcts = poss_scalars[:num_possible, 1] / 100.0  # white win prob 0-1
        d_pcts = poss_scalars[:num_possible, 2] / 100.0  # draw prob 0-1
        l_pcts = poss_scalars[:num_possible, 3] / 100.0  # black win prob 0-1

        if is_white:
            stm_evs = w_pcts + 0.5 * d_pcts
        else:
            stm_evs = l_pcts + 0.5 * d_pcts

        best_idx = int(np.argmax(stm_evs))
        best_ev = stm_evs[best_idx]
        drops = best_ev - stm_evs  # EV drop from best, per move

        # Per-move flags
        for i in range(num_possible):
            poss_scalars[i, 11] = 1.0 if drops[i] > 0.25 else 0.0   # is_mistake_move
            poss_scalars[i, 12] = 1.0 if (drops[i] < 0.025 or i == best_idx) else 0.0  # is_excellent_move

        frac_mistake_moves = float(poss_scalars[:num_possible, 11].sum()) / num_possible
        frac_excellent_moves = float(poss_scalars[:num_possible, 12].sum()) / num_possible

        # Is the actually played move a mistake?
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
        prev_was_capture,
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

    if with_phase:
        game_phase = 1  # default middlegame
        try:
            ply = int(_safe_float(move_dict.get('move_no'), 0))
            if ply <= 10:
                # Opening: first 10 half-moves (5 full moves)
                game_phase = 0
            else:
                # Endgame detection from FEN piece placement
                pieces = fen_before.split()[0]
                white_pieces = 0  # non-pawn non-king for white
                black_pieces = 0  # non-pawn non-king for black
                num_pawns = 0
                for ch in pieces:
                    if ch in 'NBRQ':
                        white_pieces += 1
                    elif ch in 'nbrq':
                        black_pieces += 1
                    elif ch in 'Pp':
                        num_pawns += 1
                # Endgame if ANY of:
                #   1. Both sides have ≤ 2 non-pawn non-king pieces
                #   2. Fewer than 3 total non-pawn non-king pieces
                #   3. No pawns on the board
                total_pieces = white_pieces + black_pieces
                if ((white_pieces <= 2 and black_pieces <= 2)
                        or total_pieces < 3
                        or num_pawns == 0):
                    game_phase = 2
        except Exception:
            pass
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
        shard_size: int = 250_000,
        val_pct: int = 10,
        test_pct: int = 10,
        with_phase: bool = False,
        run_tag: str = "",
    ):
        self.output_dir = Path(output_dir)
        self.worker_id = worker_id
        self.max_possible = max_possible
        self.shard_size = shard_size
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.with_phase = with_phase
        self.run_tag = run_tag

        # Create split directories
        for split in ('train', 'val', 'test'):
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)

        # Per-split flat buffers and shard counters.
        self._buffers: Dict[str, List[Dict]] = {
            'train': [], 'val': [], 'test': [],
        }
        self._shard_counters: Dict[str, int] = {
            'train': 0, 'val': 0, 'test': 0,
        }
        self._lock = threading.Lock()

        # Counters
        self.games_written = 0
        self.moves_written = 0
        self.examples_written = 0
        self.possible_moves_written = 0

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

        # Commit: all examples built successfully, append to flat buffer
        with self._lock:
            self._buffers[split].extend(game_examples)
            self.games_written += 1
            self.moves_written += len(move_records)
            self.examples_written += len(game_examples)
            self.possible_moves_written += len(possible_move_records)

            # Flush if buffer exceeds shard size
            if len(self._buffers[split]) >= self.shard_size:
                self._flush_shard(split, self._buffers[split])
                self._buffers[split] = []

    def _flush_shard(self, split: str, examples: List[Dict]):
        """Write a list of examples to a compressed NPZ shard.

        Atomic write: data is written to a _tmp.npz file first, then
        os.replace()'d to the final name. This prevents corrupt partial
        NPZ files if the process crashes mid-write.
        """
        if not examples:
            return

        shard_id = self._shard_counters[split]
        prefix = f'{self.run_tag}_' if self.run_tag else ''
        shard_path = self.output_dir / split / f'{prefix}shard_w{self.worker_id:02d}_{shard_id:04d}.npz'
        tmp_path = shard_path.parent / f'{shard_path.stem}_tmp.npz'

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

        np.savez_compressed(tmp_path, **save_dict)
        os.replace(tmp_path, shard_path)

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
                    self._flush_shard(split, self._buffers[split])
                    self._buffers[split] = []

    def close(self):
        """Flush and finalize."""
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
