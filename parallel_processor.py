"""
parallel_processor.py — Unified runner that processes a PGN file through
EITHER LC0 or Stockfish depth-14 workers (one engine per game).

Architecture:
    ┌─────────────┐
    │  Main Proc   │  Reads PGN, deduplicates, dispatches
    │  (PGN reader)│
    └──────┬──────┘
           │
      shared game_queue
           │
    ┌──────▼──────┐
    │  Worker pool │  LC0 + SF workers pull from same queue
    │  (N+M procs) │  Each game goes to ONE engine only
    │  each writes │  (whichever worker is free first)
    │  parquet     │
    └──────────────┘

Each game is processed by exactly one engine. Workers compete for games
from a shared queue. Dedup tracks (game_hash, engine) so restarts skip
already-processed games regardless of which engine handled them.

For each position, ALL legal moves are evaluated in a SINGLE engine call
using multipv=218 with PerPVCounters=True. Each PV gets its own independent
search tree, giving per-candidate eval + WDL without multiple round-trips.

Speed notes:
  - One analyse(nodes=250, multipv=218) call per position with PerPVCounters=True.
  - LC0 batches all NN evals internally on the GPU in one forward pass.
  - Estimated throughput: ~3-8 games/sec total with 4 workers on RTX 4090.
  - 500K games ≈ 1-2 days.
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
from datetime import datetime
from typing import Optional, List

import chess
import chess.pgn

from parquet_writer import ParquetWriter
from batch_evaluator import SyncBatchEvaluator
from stockfish_evaluator import StockfishEvaluator


# ── Deduplication ────────────────────────────────────────────────────────────

class GameDeduplicator:
    """
    SQLite-backed game deduplication.
    If pointed at an existing db, uses the games table.
    If db does not exist, creates a fresh one with a lightweight tracking table.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._use_tracking_table = False

    def connect(self):
        is_new = not os.path.exists(self.db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")

        # Always ensure the tracking table exists
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
        self._use_tracking_table = True

        count = self._conn.execute("SELECT COUNT(*) FROM processed_games").fetchone()[0]
        print(f"[DEDUP] DB: {self.db_path} | Table: processed_games | Existing rows: {count}")

    def is_duplicate(self, game_hash: str, engine: str = None) -> bool:
        """Check if game_hash exists. If engine is None, checks any engine."""
        if engine:
            row = self._conn.execute(
                "SELECT 1 FROM processed_games WHERE game_hash = ? AND engine = ? LIMIT 1",
                (game_hash, engine),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT 1 FROM processed_games WHERE game_hash = ? LIMIT 1",
                (game_hash,),
            ).fetchone()
        return row is not None

    def mark_processed(self, game_hash: str, game_id: str, engine: str):
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO processed_games (game_hash, engine, game_id, processed_at) "
                "VALUES (?, ?, ?, ?)",
                (game_hash, engine, game_id, time.time()),
            )
            self._conn.commit()
        except Exception as e:
            print(f"[DEDUP] ERROR writing {game_hash}: {e}")

    def close(self):
        if self._conn:
            self._conn.close()


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


def _process_game_with_engine(engine, game_id, pgn_text, engine_name, engine_version, opening_cache=None):
    """
    Process a single game with any engine that has:
      - evaluate_position(board, multipv=1) for position-level eval
      - evaluate_all_legal_moves(board) for per-candidate eval

    Returns (game_dict, [move_dict], [possible_move_dict]) matching parquet_schema.py.

    For each position, ALL legal moves are evaluated in a SINGLE
    engine.analyse(nodes=250, multipv=218) call with PerPVCounters=True.
    Each PV gets its own search tree, producing ~30 possible_move rows
    per actual move with genuine per-move WDL.

    Features:
    - SHA-256 hashed player names
    - Computed winner/loser/elo diffs
    - Full move detail (from_square, to_square, piece, promotion, color)
    - WDL before/after for actual moves
    - Per-legal-move eval + WDL via single multipv=218 call in evaluate_all_legal_moves()
    - Eval caching: eval_after[N] becomes eval_before[N+1] (zero extra calls)
    - time_spent computed from clock comment deltas
    - game_to_position as running SAN move list
    - static_eval_before/after set to None (no extra engine calls)
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Failed to parse PGN")

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

        # Opening cache lookup: FEN without move counters
        _oc_key = " ".join(board.fen().split()[:4]) if opening_cache else None
        _oc_hit = opening_cache.get(_oc_key) if _oc_key else None

        if cached_eval is not None:
            eval_before_cp, mate_before, wdl_w_before, wdl_d_before, wdl_l_before = cached_eval

        # If no cache, try opening cache, then fall back to engine eval
        if cached_eval is None and not board.is_game_over():
            if _oc_hit:
                oc_eval = _oc_hit["eval"]
                eval_before_cp = float(oc_eval.score_cp) if oc_eval.score_cp is not None else None
                mate_before = float(oc_eval.score_mate) if oc_eval.score_mate is not None else None
                wdl_w_before, wdl_d_before, wdl_l_before = _extract_wdl_from_eval(oc_eval)
            else:
                try:
                    eval_before_result = engine.evaluate_position(board, multipv=1)
                    eval_before_cp = float(eval_before_result.score_cp) if eval_before_result.score_cp is not None else None
                    mate_before = float(eval_before_result.score_mate) if eval_before_result.score_mate is not None else None
                    wdl_w_before, wdl_d_before, wdl_l_before = _extract_wdl_from_eval(eval_before_result)
                except Exception:
                    pass

        w_win_b, b_win_b, draw_b = _wdl_percentages(wdl_w_before, wdl_d_before, wdl_l_before)

        # ── Enumerate ALL legal moves (opening cache or engine call) ──────
        all_legal_evals = None
        if not board.is_game_over():
            if _oc_hit:
                all_legal_evals = _oc_hit["all_moves"]
            else:
                all_legal_evals = engine.evaluate_all_legal_moves(board)

        # ── Push actual move, eval AFTER ─────────────────────────────────
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
            # Try to get eval_after from the all_legal_evals results
            # (we already evaluated this child position as part of the
            # legal-moves enumeration — reuse it instead of another call)
            reused = False
            if all_legal_evals:
                for entry in all_legal_evals:
                    if entry.get("move_uci") == move.uci():
                        # This is the child position for the actual move played.
                        # Its eval is from WHITE's perspective — exactly what we want.
                        eval_after_cp = float(entry["score_cp"]) if entry.get("score_cp") is not None else None
                        mate_after = float(entry["score_mate"]) if entry.get("score_mate") is not None else None
                        wdl_w_after = entry.get("wdl_w")
                        wdl_d_after = entry.get("wdl_d")
                        wdl_l_after = entry.get("wdl_l")
                        reused = True
                        break

            if not reused:
                try:
                    eval_after_result = engine.evaluate_position(board, multipv=1)
                    eval_after_cp = float(eval_after_result.score_cp) if eval_after_result.score_cp is not None else None
                    mate_after = float(eval_after_result.score_mate) if eval_after_result.score_mate is not None else None
                    wdl_w_after, wdl_d_after, wdl_l_after = _extract_wdl_from_eval(eval_after_result)
                except Exception:
                    pass

            cached_eval = (eval_after_cp, mate_after, wdl_w_after, wdl_d_after, wdl_l_after)

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

        # ── PossibleMoveRecords from ALL legal moves ─────────────────────
        if all_legal_evals:
            board_before = chess.Board(fen_before)

            for entry in all_legal_evals:
                pm_uci = entry.get("move_uci", "")
                pm_san = entry.get("move_san", "")
                pm_fen_after = entry.get("fen_after", "")

                # Decompose the possible move
                pm_from, pm_to, pm_piece_str, pm_promo = "", "", "", ""
                try:
                    pm_move = chess.Move.from_uci(pm_uci)
                    pm_from = chess.square_name(pm_move.from_square)
                    pm_to = chess.square_name(pm_move.to_square)
                    pm_piece = board_before.piece_at(pm_move.from_square)
                    pm_piece_str = pm_piece.symbol() if pm_piece else ""
                    pm_promo = chess.piece_name(pm_move.promotion) if pm_move.promotion else ""
                except Exception:
                    pass

                pm_score_cp = float(entry["score_cp"]) if entry.get("score_cp") is not None else None
                pm_score_mate = float(entry["score_mate"]) if entry.get("score_mate") is not None else None

                # WDL percentages (engine returns 0.0-1.0 from WHITE's perspective)
                pm_w, pm_b, pm_d = _wdl_percentages(
                    entry.get("wdl_w"),
                    entry.get("wdl_d"),
                    entry.get("wdl_l"),
                )

                pm = {
                    "game_id": game_id,
                    "move_no": move_no,
                    "move_no_pair": move_no_pair,
                    "notation": pm_san,
                    "move": pm_uci,
                    "from_square": pm_from,
                    "to_square": pm_to,
                    "piece": pm_piece_str,
                    "promotion": pm_promo,
                    "color": color,
                    "fen_before": fen_before,
                    "fen_after": pm_fen_after,
                    "eval": pm_score_cp,
                    "mate_count": pm_score_mate,
                    "white_win_perc": pm_w,
                    "black_win_perc": pm_b,
                    "draw_perc": pm_d,
                    "nodes": entry.get("nodes", 0) or 0,
                    "depth": entry.get("depth", 0) or 0,
                    "pv": "",
                    "evaluated_by": engine_name,
                    "evaluator_version": engine_version,
                }
                possible_move_records.append(pm)

        node = next_node
        ply += 1

    game_rec["ply_count"] = ply
    return game_rec, move_records, possible_move_records


# ── LC0 Worker ───────────────────────────────────────────────────────────────

# ── Opening position cache (first 3 ply) ────────────────────────────────────

def _build_opening_cache(engine, max_ply=3):
    """
    Pre-compute eval_position + evaluate_all_legal_moves for all positions
    reachable within max_ply from the starting position.
    
    Returns a dict keyed by FEN (board-only, no move counters) mapping to:
        {"eval": EvalResult, "all_moves": [dict, ...]}
    
    ~8,700 positions at 3 ply, takes ~45s with one engine.
    """
    import chess
    cache = {}
    
    def _fen_key(board):
        """FEN without halfmove/fullmove counters for position matching."""
        parts = board.fen().split()
        return " ".join(parts[:4])
    
    def _explore(board, depth):
        if depth > max_ply:
            return
        key = _fen_key(board)
        if key in cache:
            return
        
        try:
            eval_result = engine.evaluate_position(board, multipv=1)
            all_moves = engine.evaluate_all_legal_moves(board)
            cache[key] = {"eval": eval_result, "all_moves": all_moves}
        except Exception as e:
            pass  # Eval failed but still recurse into children
        
        if depth < max_ply:
            for move in board.legal_moves:
                board.push(move)
                _explore(board, depth + 1)
                board.pop()
    
    board = chess.Board()
    _explore(board, 1)
    return cache


def lc0_worker(
    worker_id, game_queue, result_queue,
    lc0_path, weights_path, backend, batch_size, nodes, multipv_nodes,
    nn_cache_size, output_dir, evaluator_version, run_timestamp,
    max_games_per_batch, opening_cache=None,
):
    """LC0 worker process. Evaluates all legal moves per position."""
    # Pin worker to a specific CPU core for L3 cache locality (7800X3D V-Cache)
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            # Pin to core worker_id (mod cpu_count to stay safe)
            core = worker_id % os.cpu_count()
            mask = 1 << core
            kernel32.SetProcessAffinityMask(handle, mask)
        else:
            os.sched_setaffinity(0, {worker_id % os.cpu_count()})
    except Exception:
        pass  # Non-fatal: run without pinning

    worker_base = os.path.join(output_dir, f"worker_{worker_id:02d}_{run_timestamp}")
    writer = ParquetWriter(
        worker_base,
        worker_id=worker_id,
        max_games_per_batch=max_games_per_batch,
        possible_moves_batch_size=100_000,
    )

    engine = SyncBatchEvaluator(
        lc0_path=lc0_path,
        weights_path=weights_path,
        backend=backend,
        batch_size=batch_size,
        nodes=nodes,
        multipv_nodes=multipv_nodes,
        nn_cache_size=nn_cache_size,
    )
    engine.start()

    games_done = 0
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    try:
        while True:
            item = game_queue.get()
            if item is None:
                break
            game_id, game_hash, pgn_text = item
            try:
                game_rec, moves, pmoves = _process_game_with_engine(
                    engine, game_id, pgn_text, "lc0", evaluator_version,
                    opening_cache=opening_cache,
                )
                
                # Check if engine returned empty results (crashed silently)
                if not moves and not pmoves:
                    raise RuntimeError("Engine returned empty results - likely crashed")
                
                writer.write_game(game_rec)
                for m in moves:
                    writer.write_move(m)
                for p in pmoves:
                    writer.write_possible_move(p)
                games_done += 1
                consecutive_errors = 0  # Reset on success
                result_queue.put({
                    "status": "ok", "engine": "lc0",
                    "worker_id": worker_id,
                    "game_id": game_id, "game_hash": game_hash,
                    "num_moves": game_rec["ply_count"],
                    "num_possible_moves": len(pmoves),
                })
            except Exception as e:
                consecutive_errors += 1
                error_msg = str(e)
                
                # Check if engine process died (covers python-chess transport errors too)
                engine_crashed = any(pat in error_msg.lower() for pat in (
                    "engine process died",
                    "event loop dead",
                    "engine returned empty results",
                    "transport is closing",
                    "broken pipe",
                    "connection reset",
                    "engine not started",
                    "failed to start lc0",
                ))
                
                if engine_crashed:
                    print(f"[Worker {worker_id}] Engine crashed: {error_msg}")
                    print(f"[Worker {worker_id}] Restarting engine (attempt {consecutive_errors})...")
                    
                    # Close old engine
                    try:
                        engine.quit()
                    except:
                        pass
                    
                    # Pause before restart
                    import time
                    time.sleep(2.0)
                    
                    # Restart engine
                    try:
                        engine = SyncBatchEvaluator(
                            lc0_path=lc0_path,
                            weights_path=weights_path,
                            backend=backend,
                            batch_size=batch_size,
                            nodes=nodes,
                            multipv_nodes=multipv_nodes,
                            nn_cache_size=nn_cache_size,
                        )
                        engine.start()
                        print(f"[Worker {worker_id}] Engine restarted successfully")
                        consecutive_errors = 0
                        
                        # Retry the game once
                        print(f"[Worker {worker_id}] Retrying game {game_id}...")
                        game_rec, moves, pmoves = _process_game_with_engine(
                            engine, game_id, pgn_text, "lc0", evaluator_version,
                            opening_cache=opening_cache,
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
                            "num_possible_moves": len(pmoves),
                        })
                        continue
                    except Exception as restart_error:
                        print(f"[Worker {worker_id}] Failed to restart engine: {restart_error}")
                
                # If we get here, game failed (even after restart attempt)
                result_queue.put({
                    "status": "error", "engine": "lc0",
                    "worker_id": worker_id,
                    "game_id": game_id, "game_hash": game_hash,
                    "error": error_msg,
                })
                
                # If too many consecutive errors, give up
                if consecutive_errors >= max_consecutive_errors:
                    print(f"[Worker {worker_id}] Too many consecutive errors ({consecutive_errors}), stopping worker")
                    break
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
    output_dir, run_timestamp,
    max_games_per_batch,
):
    """Stockfish worker process. Evaluates all legal moves per position."""
    worker_base = os.path.join(output_dir, f"worker_{worker_id:02d}_{run_timestamp}")
    writer = ParquetWriter(
        worker_base,
        worker_id=worker_id,
        max_games_per_batch=max_games_per_batch,
        possible_moves_batch_size=100_000,
    )

    engine = StockfishEvaluator(
        stockfish_path=stockfish_path,
        depth=depth,
        threads=threads,
        hash_mb=hash_mb,
    )
    engine.start()
    sf_version = engine.version

    games_done = 0
    try:
        while True:
            item = game_queue.get()
            if item is None:
                break
            game_id, game_hash, pgn_text = item
            try:
                game_rec, moves, pmoves = _process_game_with_engine(
                    engine, game_id, pgn_text, "stockfish", sf_version,
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
                    "num_possible_moves": len(pmoves),
                })
            except Exception as e:
                result_queue.put({
                    "status": "error", "engine": "stockfish",
                    "worker_id": worker_id,
                    "game_id": game_id, "game_hash": game_hash,
                    "error": str(e),
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
    lc0_multipv_nodes: int = 250,
    lc0_nn_cache_size: int = 50000,
    num_lc0_workers: int = 2,
    lc0_version: str = "791556",
    # Stockfish config
    stockfish_path: str = "",
    sf_depth: int = 14,
    sf_threads: int = 1,
    sf_hash_mb: int = 256,
    num_sf_workers: int = 2,
    # General
    max_games: int = 0,
    max_games_per_batch: int = 15000,
):
    """
    Main entry point. Reads PGN, deduplicates, fans out to LC0 + Stockfish
    worker pools, collects results. All legal moves are evaluated per position.
    """
    use_lc0 = bool(lc0_path and weights_path and num_lc0_workers > 0)
    use_sf = bool(stockfish_path and num_sf_workers > 0)

    if not use_lc0 and not use_sf:
        print("ERROR: Must specify at least one engine (LC0 or Stockfish).")
        sys.exit(1)

    dedup = GameDeduplicator(db_path)
    dedup.connect()

    # Timestamp each run to prevent accidental overwrites
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    lc0_output = os.path.join(output_dir, "lc0", lc0_version)
    sf_output = os.path.join(output_dir, "stockfish_d14", f"d{sf_depth}")

    # Single shared queue — each game goes to ONE engine, not both
    game_queue: Queue = Queue(maxsize=(num_lc0_workers + num_sf_workers) * 16)
    result_queue: Queue = Queue()

    all_workers: List[Process] = []
    total_worker_count = 0

    # Spawn LC0 workers
    opening_cache = None
    if use_lc0:
        os.makedirs(lc0_output, exist_ok=True)
        # Pre-compute opening position cache (3 ply deep)
        print("[MAIN] Building opening position cache (3 ply)...")
        try:
            cache_engine = SyncBatchEvaluator(
                lc0_path=lc0_path,
                weights_path=weights_path,
                backend=backend,
                batch_size=lc0_batch_size,
                nodes=lc0_nodes,
                multipv_nodes=lc0_multipv_nodes,
                nn_cache_size=lc0_nn_cache_size,
            )
            cache_engine.start()
            opening_cache = _build_opening_cache(cache_engine, max_ply=3)
            cache_engine.quit()
            print(f"[MAIN] Opening cache: {len(opening_cache)} positions cached")
        except Exception as e:
            print(f"[MAIN] Opening cache failed ({e}), continuing without cache")
            opening_cache = None
        for wid in range(num_lc0_workers):
            p = Process(
                target=lc0_worker,
                args=(
                    wid, game_queue, result_queue,
                    lc0_path, weights_path, backend,
                    lc0_batch_size, lc0_nodes, lc0_multipv_nodes,
                    lc0_nn_cache_size, lc0_output, lc0_version, run_timestamp,
                    max_games_per_batch, opening_cache,
                ),
                daemon=True,
            )
            p.start()
            all_workers.append(p)
            total_worker_count += 1
            # Stagger worker starts to avoid GPU contention during model load
            if wid < num_lc0_workers - 1:
                import time
                time.sleep(1.0)
        print(f"[MAIN] Spawned {num_lc0_workers} LC0 workers -> {lc0_output}")
        print(f"[MAIN] Mode: ALL legal moves via multipv=218, nodes={lc0_multipv_nodes}")
        print(f"[MAIN] Batch rotation: {max_games_per_batch} games per batch dir")

    # Spawn Stockfish workers
    if use_sf:
        os.makedirs(sf_output, exist_ok=True)
        for wid in range(num_sf_workers):
            p = Process(
                target=stockfish_worker,
                args=(
                    wid, game_queue, result_queue,
                    stockfish_path, sf_depth, sf_threads, sf_hash_mb,
                    sf_output, run_timestamp,
                    max_games_per_batch,
                ),
                daemon=True,
            )
            p.start()
            all_workers.append(p)
            total_worker_count += 1
        print(f"[MAIN] Spawned {num_sf_workers} Stockfish d{sf_depth} workers -> {sf_output}")

    # Read PGN and dispatch — each game goes to ONE engine (whichever worker is free)
    dispatched = 0
    skipped = 0
    game_number = 0
    t0 = time.time()

    print(f"[MAIN] Reading PGN: {pgn_path}")
    print(f"[MAIN] Max games: {'unlimited' if max_games <= 0 else max_games}")
    print(f"[MAIN] Mode: single-engine per game (shared queue)")

    # Result collector runs in a background thread so dedup writes happen
    # in real-time while the main thread is still dispatching games.
    collector_state = {
        "lc0_completed": 0,
        "sf_completed": 0,
        "errors": 0,
        "workers_done": 0,
        "total_possible_moves": 0,
    }
    collector_done = threading.Event()

    def result_collector():
        """Drain result_queue in background, write dedup entries immediately."""
        # Own connection for this thread (SQLite objects are thread-bound)
        thread_dedup = GameDeduplicator(db_path)
        thread_dedup.connect()

        while True:
            try:
                msg = result_queue.get(timeout=5)
            except Exception:
                if collector_done.is_set():
                    thread_dedup.close()
                    break
                continue

            if msg["status"] == "worker_done":
                collector_state["workers_done"] += 1
                eng = msg["engine"]
                print(
                    f"[{eng.upper()}] Worker {msg['worker_id']} finished: "
                    f"{msg['games_done']} games, "
                    f"{msg['moves_written']} moves, "
                    f"{msg['possible_moves_written']} possible_moves written"
                )
                if collector_state["workers_done"] >= total_worker_count:
                    thread_dedup.close()
                    break
            elif msg["status"] == "ok":
                thread_dedup.mark_processed(msg["game_hash"], msg["game_id"], msg["engine"])
                collector_state["total_possible_moves"] += msg.get("num_possible_moves", 0)
                if msg["engine"] == "lc0":
                    collector_state["lc0_completed"] += 1
                else:
                    collector_state["sf_completed"] += 1
                total_done = collector_state["lc0_completed"] + collector_state["sf_completed"]
                if total_done % 20 == 0:
                    elapsed = time.time() - t0
                    rate = total_done / elapsed if elapsed > 0 else 0
                    print(
                        f"[MAIN] LC0: {collector_state['lc0_completed']} | "
                        f"SF: {collector_state['sf_completed']} | "
                        f"Total: {total_done}/{dispatched} | "
                        f"PossibleMoves: {collector_state['total_possible_moves']:,} | "
                        f"{elapsed:.1f}s ({rate:.2f} g/s)"
                    )
            elif msg["status"] == "error":
                collector_state["errors"] += 1
                print(f"[{msg['engine'].upper()}] Error: {msg.get('error', 'unknown')}")

    collector_thread = threading.Thread(target=result_collector, daemon=True)
    collector_thread.start()

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

            # Skip if already processed by ANY engine
            if dedup.is_duplicate(game_hash):
                skipped += 1
                continue

            game_queue.put((game_id, game_hash, pgn_text))
            dispatched += 1

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

    # Send poison pills — one per worker
    for _ in range(total_worker_count):
        game_queue.put(None)

    print(f"\n[MAIN] All {dispatched} games dispatched. Waiting for workers to finish...")

    # Wait for collector thread to drain all results
    collector_done.set()
    collector_thread.join(timeout=3600)

    for p in all_workers:
        p.join(timeout=15)

    lc0_completed = collector_state["lc0_completed"]
    sf_completed = collector_state["sf_completed"]
    errors = collector_state["errors"]

    dedup.close()
    elapsed = time.time() - t0

    total_done = lc0_completed + sf_completed
    rate = total_done / elapsed if elapsed > 0 else 0

    print(f"\n{'='*60}")
    print(f"  PARALLEL PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  PGN games scanned:    {game_number}")
    print(f"  Games dispatched:     {dispatched}")
    print(f"  LC0 completed:        {lc0_completed}")
    print(f"  Stockfish completed:  {sf_completed}")
    print(f"  Total possible moves: {collector_state['total_possible_moves']:,}")
    print(f"  Errors:               {errors}")
    print(f"  Wall time:            {elapsed:.1f}s ({rate:.2f} g/s)")
    print(f"  LC0 output:           {lc0_output}")
    if use_sf:
        print(f"  Stockfish output:     {sf_output}")
    print(f"  Batch size:           {max_games_per_batch} games per batch dir")
    print(f"{'='*60}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel chess game processor — LC0 + Stockfish (all legal moves)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # LC0 only, 4 workers, all legal moves evaluated:
  python parallel_processor.py games.pgn --lc0 lc0.exe --weights 791556.pb.gz --lc0-workers 4

  # Stockfish only, 8 workers at depth 14:
  python parallel_processor.py games.pgn --stockfish stockfish.exe --sf-workers 8

  # Both engines in parallel:
  python parallel_processor.py games.pgn \\
      --lc0 lc0.exe --weights 791556.pb.gz --lc0-workers 2 \\
      --stockfish stockfish.exe --sf-workers 4

  # Custom batch size (games per batch directory):
  python parallel_processor.py games.pgn --max-games-per-batch 10000 \\
      --lc0 lc0.exe --weights 791556.pb.gz

  # Increase LC0 GPU batch size for better throughput:
  python parallel_processor.py games.pgn --lc0-batch 256 \\
      --lc0 lc0.exe --weights 791556.pb.gz --lc0-workers 4
""",
    )

    parser.add_argument("pgn_file", help="Path to PGN file")
    parser.add_argument("--db", default="chessv3.db", help="SQLite dedup database (created fresh if doesn't exist)")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--max-games", type=int, default=0, help="Max games (0=all)")
    parser.add_argument("--max-games-per-batch", type=int, default=15000,
                        help="Max games per batch subdirectory (0=no batching)")

    # LC0
    lc0 = parser.add_argument_group("LC0")
    lc0.add_argument("--lc0", default="", help="Path to lc0 executable")
    lc0.add_argument("--weights", default="", help="LC0 weights file")
    lc0.add_argument("--backend", default="cuda-fp16", help="LC0 backend")
    lc0.add_argument("--lc0-batch", type=int, default=256,
                     help="LC0 minibatch size (higher = better GPU utilization with all-legal-moves)")
    lc0.add_argument("--lc0-nodes", type=int, default=1, help="LC0 nodes for single-position eval (eval_before)")
    lc0.add_argument("--lc0-multipv-nodes", type=int, default=250,
                     help="LC0 nodes for all-legal-moves multipv=218 call (default 250)")
    lc0.add_argument("--lc0-nn-cache-size", type=int, default=50000,
                     help="LC0 NNCache size (default 200000)")
    lc0.add_argument("--lc0-workers", type=int, default=2, help="Number of LC0 workers")
    lc0.add_argument("--lc0-version", default="791556", help="LC0 version tag")

    # Stockfish
    sf = parser.add_argument_group("Stockfish")
    sf.add_argument("--stockfish", default="", help="Path to stockfish executable")
    sf.add_argument("--sf-depth", type=int, default=14, help="Stockfish search depth")
    sf.add_argument("--sf-threads", type=int, default=1, help="Threads per SF instance")
    sf.add_argument("--sf-hash", type=int, default=256, help="Hash table MB per SF instance")
    sf.add_argument("--sf-workers", type=int, default=2, help="Number of SF workers")

    args = parser.parse_args()

    run_parallel(
        pgn_path=args.pgn_file,
        db_path=args.db,
        output_dir=args.output,
        lc0_path=args.lc0,
        weights_path=args.weights,
        backend=args.backend,
        lc0_batch_size=args.lc0_batch,
        lc0_nodes=args.lc0_nodes,
        lc0_multipv_nodes=args.lc0_multipv_nodes,
        lc0_nn_cache_size=args.lc0_nn_cache_size,
        num_lc0_workers=args.lc0_workers,
        lc0_version=args.lc0_version,
        stockfish_path=args.stockfish,
        sf_depth=args.sf_depth,
        sf_threads=args.sf_threads,
        sf_hash_mb=args.sf_hash,
        num_sf_workers=args.sf_workers,
        max_games=args.max_games,
        max_games_per_batch=args.max_games_per_batch,
    )


if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows multiprocessing
    main()
