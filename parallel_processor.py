"""
parallel_processor.py — Unified runner with GPU validation, dedup preload, and metrics

Enhanced with:
- LC0 GPU backend validation and smoke test
- Dedup preload (no per-game SQLite queries)
- Performance metrics collection (--profile)
- Path validation before spawn
- Auto-worker count suggestions
- Robust error handling
- Queue depth monitoring

Architecture remains: single shared queue, one engine per game.
"""

import os
import io
import sys
import time
import re
import signal
import hashlib
import sqlite3
import argparse
import threading
import multiprocessing as mp
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional, List, Set, Tuple

import chess
import chess.pgn

from parquet_writer import ParquetWriter
from batch_evaluator import SyncBatchEvaluator
from stockfish_evaluator import StockfishEvaluator
from metrics import MetricsCollector


# ── Deduplication with preload ──────────────────────────────────────────────

class GameDeduplicator:
    """
    SQLite-backed game deduplication with in-memory preload.

    At startup, loads all existing hashes into memory to avoid per-game queries.
    New hashes are batched and flushed periodically.
    """

    def __init__(self, db_path: str, batch_size: int = 10000):
        self.db_path = db_path
        self.batch_size = batch_size
        self._conn: Optional[sqlite3.Connection] = None
        self._seen: Set[Tuple[str, str]] = set()  # (game_hash, engine)
        self._pending: List[Tuple[str, str, str, float]] = []  # (hash, engine, game_id, time)
        self._lock = threading.Lock()

    def connect(self):
        """Connect to DB and preload existing hashes."""
        is_new = not os.path.exists(self.db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")

        # Ensure table exists
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_games (
                game_hash TEXT NOT NULL,
                engine TEXT NOT NULL,
                game_id TEXT,
                processed_at REAL,
                PRIMARY KEY (game_hash, engine)
            )
        """)
        self._conn.commit()

        # Preload existing hashes
        print(f"[DEDUP] Loading existing hashes from {self.db_path}...", file=sys.stderr)
        cursor = self._conn.execute("SELECT game_hash, engine FROM processed_games")
        count = 0
        for row in cursor:
            self._seen.add((row[0], row[1]))
            count += 1

        print(f"[DEDUP] Preloaded {count} existing hashes into memory", file=sys.stderr)

    def is_duplicate(self, game_hash: str, engine: str = None) -> bool:
        """Check if game_hash exists (in-memory, no DB query)."""
        with self._lock:
            if engine:
                return (game_hash, engine) in self._seen
            else:
                # Check if hash exists for any engine
                return any(h == game_hash for h, e in self._seen)

    def mark_processed(self, game_hash: str, game_id: str, engine: str):
        """Mark game as processed (in-memory + batch to DB)."""
        with self._lock:
            key = (game_hash, engine)
            if key in self._seen:
                return  # Already marked

            self._seen.add(key)
            self._pending.append((game_hash, engine, game_id, time.time()))

            # Flush batch if full
            if len(self._pending) >= self.batch_size:
                self._flush_batch()

    def _flush_batch(self):
        """Write pending hashes to DB."""
        if not self._pending:
            return

        try:
            self._conn.executemany(
                "INSERT OR IGNORE INTO processed_games (game_hash, engine, game_id, processed_at) "
                "VALUES (?, ?, ?, ?)",
                self._pending
            )
            self._conn.commit()
            flushed = len(self._pending)
            self._pending.clear()
            print(f"[DEDUP] Flushed {flushed} hashes to DB", file=sys.stderr)
        except Exception as e:
            print(f"[DEDUP] ERROR flushing batch: {e}", file=sys.stderr)

    def close(self):
        """Flush remaining hashes and close DB."""
        with self._lock:
            self._flush_batch()
            if self._conn:
                self._conn.close()
                self._conn = None


def compute_game_hash(pgn_text: str) -> str:
    return hashlib.sha256(pgn_text.strip().encode("utf-8")).hexdigest()[:16]


# ── Shared game processing logic ────────────────────────────────────────────

def _safe_int(val: str) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _hash_name(name: str) -> str:
    """SHA-256 hash a player name for anonymization."""
    if not name:
        return ""
    return hashlib.sha256(name.strip().encode("utf-8")).hexdigest()


def _parse_clock(comment: str):
    """Extract clock time from move comment like [%clk 0:05:23]."""
    m = re.search(r'\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]', comment)
    if m:
        h, mi, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return h * 3600 + mi * 60 + s
    return None


def _wdl_percentages(wdl_w, wdl_d, wdl_l):
    """
    Convert WDL values to (white_win%, black_win%, draw%) as 0-100.
    Evaluators return 0.0-1.0 values (already divided by 1000).
    """
    if wdl_w is None or wdl_d is None or wdl_l is None:
        return None, None, None
    total = wdl_w + wdl_d + wdl_l
    if total == 0:
        return None, None, None
    return (
        wdl_w / total * 100.0,   # white win %
        wdl_l / total * 100.0,   # black win %
        wdl_d / total * 100.0,   # draw %
    )


def _extract_wdl_from_eval(eval_result):
    """
    Pull WDL from an EvalResult/StockfishEvalResult.
    Returns (wdl_w, wdl_d, wdl_l) raw values, or (None, None, None).
    """
    # Direct fields on the result object
    w = getattr(eval_result, 'wdl_w', None)
    d = getattr(eval_result, 'wdl_d', None)
    l = getattr(eval_result, 'wdl_l', None)
    if w is not None and d is not None and l is not None:
        return w, d, l

    # Fall back to top PV entry
    multipv = getattr(eval_result, 'multipv', None)
    if multipv and len(multipv) > 0:
        top = multipv[0]
        w = top.get("wdl_w")
        d = top.get("wdl_d")
        l = top.get("wdl_l")
        if w is not None and d is not None and l is not None:
            return w, d, l

    return None, None, None


def _process_game_with_engine(engine, game_id, pgn_text, multipv, engine_name, engine_version, metrics=None, worker_id=None):
    """
    Process a single game with any engine that has .evaluate_position(board, multipv).
    Returns (game_dict, [move_dict], [possible_move_dict]) matching parquet_schema.py.

    Features:
    - SHA-256 hashed player names
    - Computed winner/loser/elo diffs
    - Full move detail (from_square, to_square, piece, promotion, color)
    - WDL before/after for moves
    - WDL for possible moves (from multipv)
    - Eval caching: eval_after[N] becomes eval_before[N+1] (zero extra calls)
    - time_spent computed from clock comment deltas
    - game_to_position as running SAN move list
    - static_eval_before/after set to None (no extra engine calls)
    """
    t0 = time.time()

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Failed to parse PGN")

    parse_time = time.time() - t0
    if metrics:
        metrics.record_stage_time('parse', parse_time * 1000)

    headers = game.headers
    board = game.board()

    # Hash player names
    white_raw = headers.get("White", "")
    black_raw = headers.get("Black", "")
    white_hash = _hash_name(white_raw)
    black_hash = _hash_name(black_raw)

    result_str = headers.get("Result", "*")
    white_elo = _safe_int(headers.get("WhiteElo", "0"))
    black_elo = _safe_int(headers.get("BlackElo", "0"))
    white_rating_diff = _safe_int(headers.get("WhiteRatingDiff", "0")) if headers.get("WhiteRatingDiff") else None
    black_rating_diff = _safe_int(headers.get("BlackRatingDiff", "0")) if headers.get("BlackRatingDiff") else None

    # Determine winner/loser
    winner_hash, winner_elo_val = None, None
    loser_hash, loser_elo_val = None, None
    winner_loser_elo_diff = None
    if result_str == "1-0":
        winner_hash, winner_elo_val = white_hash, white_elo
        loser_hash, loser_elo_val = black_hash, black_elo
        if white_elo and black_elo:
            winner_loser_elo_diff = white_elo - black_elo
    elif result_str == "0-1":
        winner_hash, winner_elo_val = black_hash, black_elo
        loser_hash, loser_elo_val = white_hash, white_elo
        if white_elo and black_elo:
            winner_loser_elo_diff = black_elo - white_elo

    game_hash = compute_game_hash(pgn_text)
    evaluated_at = str(time.time())

    # Count plies
    ply_count = 0
    tmp_node = game
    while tmp_node.variations:
        tmp_node = tmp_node.variation(0)
        ply_count += 1

    game_rec = {
        "game_id": game_id,
        "game_order": None,
        "event": headers.get("Event", ""),
        "site": headers.get("Site", ""),
        "date_played": headers.get("Date", ""),
        "round": headers.get("Round", ""),
        "white": white_hash,
        "black": black_hash,
        "result": result_str,
        "white_elo": white_elo,
        "white_rating_diff": white_rating_diff,
        "black_elo": black_elo,
        "black_rating_diff": black_rating_diff,
        "white_title": headers.get("WhiteTitle"),
        "black_title": headers.get("BlackTitle"),
        "winner": winner_hash,
        "winner_elo": winner_elo_val,
        "loser": loser_hash,
        "loser_elo": loser_elo_val,
        "winner_loser_elo_diff": winner_loser_elo_diff,
        "eco": headers.get("ECO", ""),
        "termination": headers.get("Termination"),
        "time_control": headers.get("TimeControl", ""),
        "utc_date": headers.get("UTCDate"),
        "utc_time": headers.get("UTCTime"),
        "variant": headers.get("Variant"),
        "ply_count": ply_count,
        "game_hash": game_hash,
        "evaluated_by": engine_name,
        "evaluator_version": engine_version,
        "evaluated_at": evaluated_at,
        "pgn_text": pgn_text,
    }

    move_records = []
    possible_move_records = []
    node = game
    ply = 0
    prev_time_white = None
    prev_time_black = None
    cached_eval = None       # (eval_cp, mate_count, wdl_w, wdl_d, wdl_l)
    san_parts = []           # Running SAN move list for game_to_position

    eval_time_total = 0.0

    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        fen_before = board.fen()

        color = "white" if board.turn == chess.WHITE else "black"
        player_hash = white_hash if board.turn == chess.WHITE else black_hash
        move_no = ply
        move_no_pair = (ply // 2) + 1

        # ── Move decomposition ───────────────────────────────────────────
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        piece_obj = board.piece_at(move.from_square)
        piece_str = piece_obj.symbol() if piece_obj else ""
        promotion_str = chess.piece_name(move.promotion) if move.promotion else ""

        try:
            notation = board.san(move)
        except Exception:
            notation = move.uci()

        # ── Clock / time_spent ───────────────────────────────────────────
        comment = next_node.comment or ""
        time_remaining = _parse_clock(comment)
        time_spent = None
        if color == "white" and time_remaining is not None and prev_time_white is not None:
            time_spent = prev_time_white - time_remaining
            if time_spent < 0:
                time_spent = None
        elif color == "black" and time_remaining is not None and prev_time_black is not None:
            time_spent = prev_time_black - time_remaining
            if time_spent < 0:
                time_spent = None

        if color == "white" and time_remaining is not None:
            prev_time_white = time_remaining
        elif color == "black" and time_remaining is not None:
            prev_time_black = time_remaining

        game_to_position = " ".join(san_parts)

        # ── Eval BEFORE (use cache from previous move's eval_after) ──────
        eval_before_cp = None
        mate_before = None
        wdl_w_before, wdl_d_before, wdl_l_before = None, None, None
        eval_result = None

        if cached_eval is not None:
            eval_before_cp, mate_before, wdl_w_before, wdl_d_before, wdl_l_before = cached_eval

        # Always run full eval for this position to get multipv for possible_moves
        if not board.is_game_over():
            t_eval = time.time()
            try:
                eval_result = engine.evaluate_position(board, multipv=multipv)
                eval_time_total += time.time() - t_eval

                # If no cache, use this eval as eval_before
                if cached_eval is None:
                    eval_before_cp = float(eval_result.score_cp) if eval_result.score_cp is not None else None
                    mate_before = float(eval_result.score_mate) if eval_result.score_mate is not None else None
                    wdl_w_before, wdl_d_before, wdl_l_before = _extract_wdl_from_eval(eval_result)
            except Exception as e:
                # Engine failed — skip this move's eval but still record the move
                if metrics and worker_id is not None:
                    metrics.worker_idle(worker_id)
                print(f"[{engine_name}] Eval failed at ply {ply}: {e}", file=sys.stderr)
        else:
            # Terminal position before move (shouldn't happen in valid PGN)
            cached_eval = None

        w_win_b, b_win_b, draw_b = _wdl_percentages(wdl_w_before, wdl_d_before, wdl_l_before)

        # ── Push move, eval AFTER ────────────────────────────────────────
        board.push(move)
        fen_after = board.fen()
        san_parts.append(notation)

        eval_after_cp = None
        mate_after = None
        wdl_w_after, wdl_d_after, wdl_l_after = None, None, None

        if board.is_game_over():
            # Terminal — derive eval from outcome
            if board.is_checkmate():
                # Side to move is mated
                eval_after_cp = -10000.0 if board.turn == chess.WHITE else 10000.0
                mate_after = 0.0
                if board.turn == chess.WHITE:
                    wdl_w_after, wdl_d_after, wdl_l_after = 0.0, 0.0, 1.0
                else:
                    wdl_w_after, wdl_d_after, wdl_l_after = 1.0, 0.0, 0.0
            else:
                eval_after_cp = 0.0
                mate_after = None
                wdl_w_after, wdl_d_after, wdl_l_after = 0.0, 1.0, 0.0
            cached_eval = (eval_after_cp, mate_after, wdl_w_after, wdl_d_after, wdl_l_after)
        else:
            t_eval = time.time()
            try:
                eval_after_result = engine.evaluate_position(board, multipv=1)
                eval_time_total += time.time() - t_eval

                eval_after_cp = float(eval_after_result.score_cp) if eval_after_result.score_cp is not None else None
                mate_after = float(eval_after_result.score_mate) if eval_after_result.score_mate is not None else None
                wdl_w_after, wdl_d_after, wdl_l_after = _extract_wdl_from_eval(eval_after_result)
                cached_eval = (eval_after_cp, mate_after, wdl_w_after, wdl_d_after, wdl_l_after)
            except Exception as e:
                if metrics and worker_id is not None:
                    metrics.worker_idle(worker_id)
                print(f"[{engine_name}] Eval after failed at ply {ply}: {e}", file=sys.stderr)
                cached_eval = None

        w_win_a, b_win_a, draw_a = _wdl_percentages(wdl_w_after, wdl_d_after, wdl_l_after)

        # ── MoveRecord ───────────────────────────────────────────────────
        mr = {
            "game_id": game_id,
            "move_no": move_no,
            "move_no_pair": move_no_pair,
            "player": player_hash,
            "notation": notation,
            "move": move.uci(),
            "from_square": from_sq,
            "to_square": to_sq,
            "piece": piece_str,
            "promotion": promotion_str,
            "color": color,
            "fen_before": fen_before,
            "fen_after": fen_after,
            "time_remaining": time_remaining,
            "time_spent": time_spent,
            "game_to_position": game_to_position,
            "white_win_perc_before": w_win_b,
            "black_win_perc_before": b_win_b,
            "draw_perc_before": draw_b,
            "white_win_perc_after": w_win_a,
            "black_win_perc_after": b_win_a,
            "draw_perc_after": draw_a,
            "static_eval_before": None,
            "static_eval_after": None,
            "eval_before": eval_before_cp,
            "mate_count_before": mate_before,
            "eval_after": eval_after_cp,
            "mate_count_after": mate_after,
            "evaluated_by": engine_name,
            "evaluator_version": engine_version,
        }
        move_records.append(mr)

        # ── PossibleMoveRecords from multipv ─────────────────────────────
        if eval_result and eval_result.multipv:
            board_before = chess.Board(fen_before)

            for pv_entry in eval_result.multipv:
                pv_uci = pv_entry.get("move_uci", "")
                pv_san = pv_entry.get("move_san", "")
                pv_eval = float(pv_entry["score_cp"]) if pv_entry.get("score_cp") is not None else None
                pv_mate = float(pv_entry["score_mate"]) if pv_entry.get("score_mate") is not None else None
                pv_nodes = pv_entry.get("nodes", 0)
                pv_depth = pv_entry.get("depth", 0)

                # Decompose the possible move
                pm_from, pm_to, pm_piece_str, pm_promo, pm_fen_after = "", "", "", "", ""
                try:
                    pm_move = chess.Move.from_uci(pv_uci)
                    pm_from = chess.square_name(pm_move.from_square)
                    pm_to = chess.square_name(pm_move.to_square)
                    pm_piece = board_before.piece_at(pm_move.from_square)
                    pm_piece_str = pm_piece.symbol() if pm_piece else ""
                    pm_promo = chess.piece_name(pm_move.promotion) if pm_move.promotion else ""
                    board_copy = board_before.copy()
                    board_copy.push(pm_move)
                    pm_fen_after = board_copy.fen()
                except Exception:
                    pass

                # Per-PV WDL (fall back to top-level eval WDL if per-PV missing)
                pv_wdl_w = pv_entry.get("wdl_w")
                pv_wdl_d = pv_entry.get("wdl_d")
                pv_wdl_l = pv_entry.get("wdl_l")
                if pv_wdl_w is None and eval_result:
                    pv_wdl_w = getattr(eval_result, 'wdl_w', None)
                    pv_wdl_d = getattr(eval_result, 'wdl_d', None)
                    pv_wdl_l = getattr(eval_result, 'wdl_l', None)
                pm_w, pm_b, pm_d = _wdl_percentages(pv_wdl_w, pv_wdl_d, pv_wdl_l)

                pm = {
                    "game_id": game_id,
                    "move_no": move_no,
                    "move_no_pair": move_no_pair,
                    "notation": pv_san,
                    "move": pv_uci,
                    "from_square": pm_from,
                    "to_square": pm_to,
                    "piece": pm_piece_str,
                    "promotion": pm_promo,
                    "color": color,
                    "fen_before": fen_before,
                    "fen_after": pm_fen_after,
                    "eval": pv_eval,
                    "mate_count": pv_mate,
                    "white_win_perc": pm_w,
                    "black_win_perc": pm_b,
                    "draw_perc": pm_d,
                    "nodes": pv_nodes or 0,
                    "depth": pv_depth or 0,
                    "pv": "",
                    "evaluated_by": engine_name,
                    "evaluator_version": engine_version,
                }
                possible_move_records.append(pm)

        node = next_node
        ply += 1

    game_rec["ply_count"] = ply

    total_time = time.time() - t0
    if metrics:
        metrics.record_stage_time('eval', eval_time_total * 1000)
        metrics.record_stage_time('total_game', total_time * 1000)

    return game_rec, move_records, possible_move_records


# ── LC0 Worker ───────────────────────────────────────────────────────────────

def lc0_worker(
    worker_id, game_queue, result_queue,
    lc0_path, weights_path, backend, batch_size, nodes,
    output_dir, multipv, evaluator_version,
    verify_gpu=True, metrics=None,
):
    """LC0 worker process."""
    if metrics:
        metrics.register_worker(worker_id, 'lc0')

    worker_output = os.path.join(output_dir, f"worker_{worker_id:02d}")
    writer = ParquetWriter(worker_output, worker_id=worker_id)

    # Validate paths before spawn
    if not os.path.exists(lc0_path):
        result_queue.put({
            "status": "error", "engine": "lc0", "worker_id": worker_id,
            "error": f"LC0 executable not found: {lc0_path}"
        })
        return

    if not os.path.exists(weights_path):
        result_queue.put({
            "status": "error", "engine": "lc0", "worker_id": worker_id,
            "error": f"Weights file not found: {weights_path}"
        })
        return

    engine = SyncBatchEvaluator(
        lc0_path=lc0_path,
        weights_path=weights_path,
        backend=backend,
        batch_size=batch_size,
        nodes=nodes,
        verify_gpu=verify_gpu,
    )

    try:
        engine.start()
    except Exception as e:
        result_queue.put({
            "status": "error", "engine": "lc0", "worker_id": worker_id,
            "error": f"Failed to start LC0: {e}"
        })
        return

    games_done = 0
    try:
        while True:
            if metrics:
                metrics.worker_idle(worker_id)

            item = game_queue.get()
            if item is None:
                break

            if metrics:
                metrics.worker_busy(worker_id)

            game_id, game_hash, pgn_text = item
            try:
                game_rec, moves, pmoves = _process_game_with_engine(
                    engine, game_id, pgn_text, multipv, "lc0", evaluator_version,
                    metrics=metrics, worker_id=worker_id,
                )
                writer.write_game(game_rec)
                for m in moves:
                    writer.write_move(m)
                for p in pmoves:
                    writer.write_possible_move(p)
                games_done += 1
                result_queue.put({
                    "status": "ok", "engine": "lc0",
                    "worker_id": worker_id,
                    "game_id": game_id, "game_hash": game_hash,
                    "num_moves": game_rec["ply_count"],
                })
            except Exception as e:
                import traceback
                result_queue.put({
                    "status": "error", "engine": "lc0",
                    "worker_id": worker_id,
                    "game_id": game_id, "game_hash": game_hash,
                    "error": f"{e}\n{traceback.format_exc()[:500]}",
                })
    finally:
        writer.close()
        engine.quit()
        result_queue.put({
            "status": "worker_done", "engine": "lc0",
            "worker_id": worker_id, "games_done": games_done,
            "games_written": writer.games_written,
            "moves_written": writer.moves_written,
            "possible_moves_written": writer.possible_moves_written,
        })


# ── Stockfish Worker ─────────────────────────────────────────────────────────

def stockfish_worker(
    worker_id, game_queue, result_queue,
    stockfish_path, depth, threads, hash_mb,
    output_dir, multipv, metrics=None,
):
    """Stockfish depth-14 worker process."""
    if metrics:
        metrics.register_worker(worker_id + 1000, 'stockfish')  # Offset IDs to avoid collision

    worker_output = os.path.join(output_dir, f"worker_{worker_id:02d}")
    writer = ParquetWriter(worker_output, worker_id=worker_id)

    # Validate path
    if not os.path.exists(stockfish_path):
        result_queue.put({
            "status": "error", "engine": "stockfish", "worker_id": worker_id,
            "error": f"Stockfish executable not found: {stockfish_path}"
        })
        return

    engine = StockfishEvaluator(
        stockfish_path=stockfish_path,
        depth=depth,
        threads=threads,
        hash_mb=hash_mb,
    )

    try:
        engine.start()
    except Exception as e:
        result_queue.put({
            "status": "error", "engine": "stockfish", "worker_id": worker_id,
            "error": f"Failed to start Stockfish: {e}"
        })
        return

    sf_version = engine.version
    games_done = 0

    try:
        while True:
            if metrics:
                metrics.worker_idle(worker_id + 1000)

            item = game_queue.get()
            if item is None:
                break

            if metrics:
                metrics.worker_busy(worker_id + 1000)

            game_id, game_hash, pgn_text = item
            try:
                game_rec, moves, pmoves = _process_game_with_engine(
                    engine, game_id, pgn_text, multipv, "stockfish", sf_version,
                    metrics=metrics, worker_id=worker_id + 1000,
                )
                writer.write_game(game_rec)
                for m in moves:
                    writer.write_move(m)
                for p in pmoves:
                    writer.write_possible_move(p)
                games_done += 1
                result_queue.put({
                    "status": "ok", "engine": "stockfish",
                    "worker_id": worker_id,
                    "game_id": game_id, "game_hash": game_hash,
                    "num_moves": game_rec["ply_count"],
                })
            except Exception as e:
                import traceback
                result_queue.put({
                    "status": "error", "engine": "stockfish",
                    "worker_id": worker_id,
                    "game_id": game_id, "game_hash": game_hash,
                    "error": f"{e}\n{traceback.format_exc()[:500]}",
                })
    finally:
        writer.close()
        engine.quit()
        result_queue.put({
            "status": "worker_done", "engine": "stockfish",
            "worker_id": worker_id, "games_done": games_done,
            "games_written": writer.games_written,
            "moves_written": writer.moves_written,
            "possible_moves_written": writer.possible_moves_written,
        })


# ── Main orchestrator ────────────────────────────────────────────────────────

def run_parallel(
    pgn_path: str,
    db_path: str = "games.db",
    output_dir: str = "output",
    # LC0 config
    lc0_path: str = "",
    weights_path: str = "",
    backend: str = "cuda-fp16",
    lc0_batch_size: int = 32,
    lc0_nodes: int = 1,
    num_lc0_workers: int = 2,
    lc0_version: str = "791556",
    verify_gpu: bool = True,
    # Stockfish config
    stockfish_path: str = "",
    sf_depth: int = 14,
    sf_threads: int = 1,
    sf_hash_mb: int = 256,
    num_sf_workers: int = 2,
    # General
    max_games: int = 0,
    multipv: int = 1,
    profile: bool = False,
):
    """
    Main entry point. Reads PGN, deduplicates, fans out to LC0 + Stockfish
    worker pools, collects results.
    """
    use_lc0 = bool(lc0_path and weights_path and num_lc0_workers > 0)
    use_sf = bool(stockfish_path and num_sf_workers > 0)

    if not use_lc0 and not use_sf:
        print("ERROR: Must specify at least one engine (LC0 or Stockfish).")
        sys.exit(1)

    # Validate paths early
    if use_lc0:
        if not os.path.exists(lc0_path):
            print(f"ERROR: LC0 executable not found: {lc0_path}")
            sys.exit(1)
        if not os.path.exists(weights_path):
            print(f"ERROR: Weights file not found: {weights_path}")
            sys.exit(1)

    if use_sf and not os.path.exists(stockfish_path):
        print(f"ERROR: Stockfish executable not found: {stockfish_path}")
        sys.exit(1)

    # Auto-tune worker counts if oversubscribed
    import os as _os
    cpu_count = _os.cpu_count() or 16
    if use_sf and num_sf_workers > cpu_count:
        print(f"WARNING: Requested {num_sf_workers} Stockfish workers but only {cpu_count} CPUs", file=sys.stderr)
        print(f"WARNING: Consider reducing --sf-workers to {cpu_count - num_lc0_workers - 2}", file=sys.stderr)

    # Initialize dedup with preload
    dedup = GameDeduplicator(db_path)
    dedup.connect()

    # Initialize metrics
    metrics = None
    if profile:
        metrics = MetricsCollector(report_interval=30.0)
        metrics.start()
        print("[METRICS] Performance profiling enabled", file=sys.stderr)

    lc0_output = os.path.join(output_dir, "lc0", lc0_version)
    sf_output = os.path.join(output_dir, "stockfish_d14", "latest")

    # Single shared queue — each game goes to ONE engine, not both
    queue_maxsize = (num_lc0_workers + num_sf_workers) * 4
    game_queue: Queue = Queue(maxsize=queue_maxsize)
    result_queue: Queue = Queue()

    if metrics:
        metrics.register_queue('game', queue_maxsize)
        metrics.register_queue('result', 0)  # Unbounded

    all_workers: List[Process] = []
    total_worker_count = 0

    # Spawn LC0 workers
    if use_lc0:
        os.makedirs(lc0_output, exist_ok=True)
        for wid in range(num_lc0_workers):
            p = Process(
                target=lc0_worker,
                args=(
                    wid, game_queue, result_queue,
                    lc0_path, weights_path, backend,
                    lc0_batch_size, lc0_nodes,
                    lc0_output, multipv, lc0_version,
                    verify_gpu, None,  # metrics can't be pickled across processes on Windows
                ),
                daemon=True,
            )
            p.start()
            all_workers.append(p)
            total_worker_count += 1
            if metrics:
                metrics.register_worker(wid, 'lc0')
        print(f"[MAIN] Spawned {num_lc0_workers} LC0 workers -> {lc0_output}")

    # Spawn Stockfish workers
    if use_sf:
        os.makedirs(sf_output, exist_ok=True)
        for wid in range(num_sf_workers):
            p = Process(
                target=stockfish_worker,
                args=(
                    wid, game_queue, result_queue,
                    stockfish_path, sf_depth, sf_threads, sf_hash_mb,
                    sf_output, multipv, None,  # metrics can't be pickled across processes on Windows
                ),
                daemon=True,
            )
            p.start()
            all_workers.append(p)
            total_worker_count += 1
            if metrics:
                metrics.register_worker(wid + 1000, 'stockfish')
        print(f"[MAIN] Spawned {num_sf_workers} Stockfish d{sf_depth} workers -> {sf_output}")

    # Read PGN and dispatch — each game goes to ONE engine (whichever worker is free)
    dispatched = 0
    skipped = 0
    game_number = 0
    t0 = time.time()

    print(f"[MAIN] Reading PGN: {pgn_path}")
    print(f"[MAIN] Max games: {'unlimited' if max_games <= 0 else max_games}")
    print(f"[MAIN] Mode: single-engine per game (shared queue)")
    print(f"[MAIN] Dedup preload: {len(dedup._seen)} hashes loaded")

    # Result collector runs in a background thread so dedup writes happen
    # in real-time while the main thread is still dispatching games.
    collector_state = {
        "lc0_completed": 0,
        "sf_completed": 0,
        "errors": 0,
        "workers_done": 0,
    }
    collector_done = threading.Event()
    queue_empty_since = None

    def result_collector():
        """Drain result_queue in background, write dedup entries immediately."""
        nonlocal queue_empty_since

        # Own connection for this thread (SQLite objects are thread-bound)
        thread_dedup = GameDeduplicator(db_path)
        thread_dedup.connect()

        while True:
            try:
                msg = result_queue.get(timeout=5)
                queue_empty_since = None
            except Exception:
                # Check if game queue is starving workers
                if game_queue.qsize() == 0:
                    if queue_empty_since is None:
                        queue_empty_since = time.time()
                    elif time.time() - queue_empty_since > 10:
                        print(f"[MAIN WARNING] Game queue empty for >10s — parser may be starved", file=sys.stderr)
                        queue_empty_since = time.time()  # Reset to avoid spam

                if collector_done.is_set():
                    thread_dedup.close()
                    break
                continue

            if metrics:
                metrics.record_queue_size('result', result_queue.qsize())

            if msg["status"] == "worker_done":
                collector_state["workers_done"] += 1
                eng = msg["engine"]
                print(
                    f"[{eng.upper()}] Worker {msg['worker_id']} finished: "
                    f"{msg['games_done']} games, "
                    f"{msg['moves_written']} moves written"
                )
                if collector_state["workers_done"] >= total_worker_count:
                    thread_dedup.close()
                    break
            elif msg["status"] == "ok":
                thread_dedup.mark_processed(msg["game_hash"], msg["game_id"], msg["engine"])
                if msg["engine"] == "lc0":
                    collector_state["lc0_completed"] += 1
                else:
                    collector_state["sf_completed"] += 1

                if metrics:
                    metrics.increment_completed()
                    metrics.record_queue_size('game', game_queue.qsize())

                total_done = collector_state["lc0_completed"] + collector_state["sf_completed"]
                if total_done % 20 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"[MAIN] LC0: {collector_state['lc0_completed']} | "
                        f"SF: {collector_state['sf_completed']} | "
                        f"Total: {total_done}/{dispatched} | "
                        f"{elapsed:.1f}s"
                    )
            elif msg["status"] == "error":
                collector_state["errors"] += 1
                print(f"[{msg['engine'].upper()}] Error in worker {msg['worker_id']}: {msg.get('error', 'unknown')}")

    collector_thread = threading.Thread(target=result_collector, daemon=True)
    collector_thread.start()

    try:
        with open(pgn_path, "r", errors="replace") as f:
            while True:
                if max_games > 0 and dispatched >= max_games:
                    break

                game = chess.pgn.read_game(f)
                if game is None:
                    break

                game_number += 1
                pgn_text = str(game)
                game_hash = compute_game_hash(pgn_text)
                game_id = f"game_{game_hash}"

                # Skip if already processed by ANY engine (in-memory check, no DB query)
                if dedup.is_duplicate(game_hash):
                    skipped += 1
                    if metrics:
                        metrics.increment_skipped()
                    continue

                # Block if queue is full (backpressure)
                game_queue.put((game_id, game_hash, pgn_text))
                dispatched += 1

                if metrics:
                    metrics.increment_dispatched()
                    metrics.record_queue_size('game', game_queue.qsize())

                if game_number % 100 == 0:
                    elapsed = time.time() - t0
                    total_done = collector_state["lc0_completed"] + collector_state["sf_completed"]
                    print(
                        f"[MAIN] Scanned {game_number} games | "
                        f"Dispatched: {dispatched} | "
                        f"Completed: {total_done} | "
                        f"Skipped: {skipped} | "
                        f"{elapsed:.1f}s"
                    )

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user, shutting down...", file=sys.stderr)

    # Send poison pills — one per worker
    for _ in range(total_worker_count):
        game_queue.put(None)

    print(f"\n[MAIN] All {dispatched} games dispatched. Waiting for workers to finish...")

    # Wait for collector thread to drain all results
    collector_done.set()
    collector_thread.join(timeout=3600)

    for p in all_workers:
        p.join(timeout=15)
        if p.is_alive():
            print(f"[MAIN WARNING] Worker {p.pid} did not exit cleanly, terminating...", file=sys.stderr)
            p.terminate()

    lc0_completed = collector_state["lc0_completed"]
    sf_completed = collector_state["sf_completed"]
    errors = collector_state["errors"]

    dedup.close()

    if metrics:
        metrics.stop()
        print(metrics.final_report())

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  PARALLEL PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  PGN games scanned:    {game_number}")
    print(f"  Games dispatched:     {dispatched}")
    print(f"  LC0 completed:        {lc0_completed}")
    print(f"  Stockfish completed:  {sf_completed}")
    print(f"  Errors:               {errors}")
    print(f"  Wall time:            {elapsed:.1f}s")
    if dispatched > 0:
        print(f"  Throughput:           {dispatched/elapsed:.2f} games/s")
    print(f"  LC0 output:           {lc0_output}")
    if use_sf:
        print(f"  Stockfish output:     {sf_output}")
    print(f"{'='*60}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel chess game processor — LC0 + Stockfish with GPU validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # LC0 only, 4 workers with GPU validation:
  python parallel_processor.py games.pgn --lc0 lc0.exe --weights 791556.pb.gz --lc0-workers 4

  # Stockfish only, 8 workers at depth 14:
  python parallel_processor.py games.pgn --stockfish stockfish.exe --sf-workers 8

  # Both engines in parallel with profiling:
  python parallel_processor.py games.pgn \\
      --lc0 lc0.exe --weights 791556.pb.gz --lc0-workers 2 \\
      --stockfish stockfish.exe --sf-workers 4 \\
      --profile

  # Limit to 100 games with auto-tune:
  python parallel_processor.py games.pgn --max-games 100 \\
      --lc0 lc0.exe --weights 791556.pb.gz \\
      --stockfish stockfish.exe

GPU Validation:
  The script validates that LC0 executable exists, weights file exists,
  and the requested backend is supported. It runs a smoke test on startup
  to verify the engine responds. If GPU is not detected, it will warn you.

Performance Tuning:
  - LC0 batch size: 32-128 (higher = more GPU utilization)
  - Stockfish workers: ~1 per CPU core (minus LC0 workers)
  - Use --profile to see bottleneck analysis

Troubleshooting:
  - If GPU util stays at 0%, check nvidia-smi during run
  - If workers are idle, parser may be bottleneck (try faster SSD)
  - If queue fills up, add more workers or reduce batch size
""",
    )

    parser.add_argument("pgn_file", help="Path to PGN file")
    parser.add_argument("--db", default="chessv3.db", help="SQLite dedup database (created fresh if doesn't exist)")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--max-games", type=int, default=0, help="Max games (0=all)")
    parser.add_argument("--multipv", type=int, default=1, help="Multi-PV lines")
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling and metrics")

    # LC0
    lc0 = parser.add_argument_group("LC0")
    lc0.add_argument("--lc0", default="", help="Path to lc0 executable")
    lc0.add_argument("--weights", default="", help="LC0 weights file")
    lc0.add_argument("--backend", default="cuda-fp16", help="LC0 backend (cuda-fp16, cuda, opencl, etc.)")
    lc0.add_argument("--lc0-batch", type=int, default=32, help="LC0 minibatch size (16/32/64/128)")
    lc0.add_argument("--lc0-nodes", type=int, default=1, help="LC0 nodes per position")
    lc0.add_argument("--lc0-workers", type=int, default=2, help="Number of LC0 workers")
    lc0.add_argument("--lc0-version", default="791556", help="LC0 version tag")
    lc0.add_argument("--no-verify-gpu", dest="verify_gpu", action="store_false", help="Skip GPU validation")

    # Stockfish
    sf = parser.add_argument_group("Stockfish")
    sf.add_argument("--stockfish", default="", help="Path to stockfish executable")
    sf.add_argument("--sf-depth", type=int, default=14, help="Stockfish search depth")
    sf.add_argument("--sf-threads", type=int, default=1, help="Threads per SF instance")
    sf.add_argument("--sf-hash", type=int, default=256, help="Hash table MB per SF instance")
    sf.add_argument("--sf-workers", type=int, default=2, help="Number of SF workers")

    args = parser.parse_args()

    # Auto-tune worker counts if oversubscribed
    cpu_count = os.cpu_count() or 16
    if args.stockfish and args.sf_workers > cpu_count:
        print(f"WARNING: {args.sf_workers} SF workers requested but only {cpu_count} CPUs available", file=sys.stderr)
        suggested = max(1, cpu_count - args.lc0_workers - 2)
        print(f"WARNING: Consider --sf-workers {suggested}", file=sys.stderr)

    run_parallel(
        pgn_path=args.pgn_file,
        db_path=args.db,
        output_dir=args.output,
        lc0_path=args.lc0,
        weights_path=args.weights,
        backend=args.backend,
        lc0_batch_size=args.lc0_batch,
        lc0_nodes=args.lc0_nodes,
        num_lc0_workers=args.lc0_workers,
        lc0_version=args.lc0_version,
        verify_gpu=args.verify_gpu,
        stockfish_path=args.stockfish,
        sf_depth=args.sf_depth,
        sf_threads=args.sf_threads,
        sf_hash_mb=args.sf_hash,
        num_sf_workers=args.sf_workers,
        max_games=args.max_games,
        multipv=args.multipv,
        profile=args.profile,
    )


if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows multiprocessing
    main()
