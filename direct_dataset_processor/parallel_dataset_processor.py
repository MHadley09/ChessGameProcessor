"""
parallel_dataset_processor.py — Unified runner that processes a PGN file
directly into NPZ dataset shards (no intermediate parquet).

Architecture:
    ┌─────────────┐
    │  Main Proc   │  Reads PGN, deduplicates, dispatches
    │  (PGN reader)│
    └──────┬──────┘
           │
      shared game_queue
           │
    ┌──────▼──────┐
    │  Worker pool │  LC0 workers pull from game queue
    │  (N procs)   │  Each game processed by LC0
    │  each writes │  Dataset examples directly to NPZ shards
    │  NPZ shards  │
    └──────────────┘

Each position is evaluated for ALL legal moves via LC0, then converted
to a MIMO dataset example and written directly to NPZ shards.

For each position, ALL legal moves are evaluated in a SINGLE engine call
using multipv=218 with PerPVCounters=True.

Output: output_dir/{train,val,test}/shard_wXX_XXXX.npz
Compatible with MIMOCompactDataset from dataset_v4.py.
"""

import os
import io
import sys
import time
import json
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

from dataset_writer import DatasetWriter
from batch_evaluator import SyncBatchEvaluator
from direct_evaluator import SyncDirectEvaluator


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
    if not name:
        return ""
    return hashlib.sha256(name.strip().encode("utf-8")).hexdigest()


def _parse_clock(comment: str):
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
    w = getattr(eval_result, 'wdl_w', None)
    d = getattr(eval_result, 'wdl_d', None)
    l = getattr(eval_result, 'wdl_l', None)
    if w is not None and d is not None and l is not None:
        return w, d, l
    multipv = getattr(eval_result, 'multipv', None)
    if multipv and len(multipv) > 0:
        top = multipv[0]
        w = top.get("wdl_w")
        d = top.get("wdl_d")
        l = top.get("wdl_l")
        if w is not None and d is not None and l is not None:
            return w, d, l
    return None, None, None



def _parse_time_control(tc_str: str) -> int:
    """Parse Lichess TimeControl header string and compute effective time.
    
    Format: "start+increment" (e.g., "180+2", "600+0", "300+3").
    Effective time = start + 30 * increment (approx. 30-move game).
    Returns 0 if unparseable.
    """
    if not tc_str or tc_str == "-":
        return 0
    try:
        parts = tc_str.split("+")
        start = int(parts[0])
        increment = int(parts[1]) if len(parts) > 1 else 0
        return start + 30 * increment
    except (ValueError, IndexError):
        return 0


def _is_bot(title: str) -> bool:
    """Check if a player title indicates a bot account."""
    if not title:
        return False
    return title.upper() == "BOT"


def _process_game_with_engine(engine, game_id, pgn_text, engine_name, engine_version,
                              min_time_control=0, skip_bots=False, humans_only=False):
    """
    Process a single game with any engine.
    Returns (game_dict, [move_dict], [possible_move_dict]).

    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Failed to parse PGN")

    headers = game.headers
    board = game.board()

    white_raw = headers.get("White", "")
    black_raw = headers.get("Black", "")
    white_hash = _hash_name(white_raw)
    black_hash = _hash_name(black_raw)

    result_str = headers.get("Result", "*")
    white_elo = _safe_int(headers.get("WhiteElo", "0"))
    black_elo = _safe_int(headers.get("BlackElo", "0"))
    white_rating_diff = _safe_int(headers.get("WhiteRatingDiff", "0")) if headers.get("WhiteRatingDiff") else None
    black_rating_diff = _safe_int(headers.get("BlackRatingDiff", "0")) if headers.get("BlackRatingDiff") else None

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
    }


    # --- Filtering ---
    tc_str = headers.get("TimeControl", "")
    effective_tc = _parse_time_control(tc_str)
    white_title = headers.get("WhiteTitle", "")
    black_title = headers.get("BlackTitle", "")
    white_is_bot = _is_bot(white_title)
    black_is_bot = _is_bot(black_title)

    if min_time_control > 0 and effective_tc < min_time_control:
        return None  # Skip: time control too short

    if skip_bots and white_is_bot and black_is_bot:
        return None  # Skip: both players are bots

    move_records = []
    possible_move_records = []
    node = game
    ply = 0
    prev_time_white = None
    prev_time_black = None
    cached_eval = None
    san_parts = []

    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        fen_before = board.fen()

        color = "white" if board.turn == chess.WHITE else "black"
        player_hash = white_hash if board.turn == chess.WHITE else black_hash

        # Human-only mode: skip moves where side-to-move is a bot
        side_is_bot = white_is_bot if board.turn == chess.WHITE else black_is_bot
        if humans_only and side_is_bot:
            # Still push the move to keep board state and SAN history correct
            try:
                skip_san = board.san(move)
            except Exception:
                skip_san = move.uci()
            board.push(move)
            san_parts.append(skip_san)
            ply += 1
            node = next_node
            continue
        move_no = ply
        move_no_pair = (ply // 2) + 1

        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        piece_obj = board.piece_at(move.from_square)
        piece_str = piece_obj.symbol() if piece_obj else ""
        promotion_str = chess.piece_name(move.promotion) if move.promotion else ""

        try:
            notation = board.san(move)
        except Exception:
            notation = move.uci()

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

        eval_before_cp = None
        mate_before = None
        wdl_w_before, wdl_d_before, wdl_l_before = None, None, None

        if cached_eval is not None:
            eval_before_cp, mate_before, wdl_w_before, wdl_d_before, wdl_l_before = cached_eval

        if cached_eval is None and not board.is_game_over():
            try:
                eval_before_result = engine.evaluate_position(board, multipv=1)
                eval_before_cp = float(eval_before_result.score_cp) if eval_before_result.score_cp is not None else None
                mate_before = float(eval_before_result.score_mate) if eval_before_result.score_mate is not None else None
                wdl_w_before, wdl_d_before, wdl_l_before = _extract_wdl_from_eval(eval_before_result)
            except Exception:
                pass

        w_win_b, b_win_b, draw_b = _wdl_percentages(wdl_w_before, wdl_d_before, wdl_l_before)

        all_legal_evals = None
        if not board.is_game_over():
            try:
                all_legal_evals = engine.evaluate_all_legal_moves(board)
            except Exception:
                all_legal_evals = None

        board.push(move)
        fen_after = board.fen()
        san_parts.append(notation)

        eval_after_cp = None
        mate_after = None
        wdl_w_after, wdl_d_after, wdl_l_after = None, None, None

        if board.is_game_over():
            if board.is_checkmate():
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
            reused = False
            if all_legal_evals:
                for entry in all_legal_evals:
                    if entry.get("move_uci") == move.uci():
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

        if all_legal_evals:
            board_before = chess.Board(fen_before)

            for entry in all_legal_evals:
                pm_uci = entry.get("move_uci", "")
                pm_san = entry.get("move_san", "")
                pm_fen_after = entry.get("fen_after", "")

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

def lc0_worker(
    worker_id, game_queue, result_queue,
    lc0_path, weights_path, backend, batch_size, min_nodes, max_nodes, nodes_mult,
    min_time_control, skip_bots, humans_only,
    output_dir, evaluator_version, run_timestamp,
    max_possible, shard_size, val_pct, test_pct, with_phase,
    use_direct_uci=True,
):
    """LC0 worker process. Evaluates all legal moves per position.
    Writes NPZ dataset shards directly instead of parquet."""
    writer = DatasetWriter(
        output_dir,
        worker_id=worker_id,
        max_possible=max_possible,
        shard_size=shard_size,
        val_pct=val_pct,
        test_pct=test_pct,
        with_phase=with_phase,
    )

    EvaluatorClass = SyncDirectEvaluator if use_direct_uci else SyncBatchEvaluator
    engine = EvaluatorClass(
        lc0_path=lc0_path,
        weights_path=weights_path,
        backend=backend,
        batch_size=batch_size,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        nodes_mult=nodes_mult,
    )
    engine.start()

    games_done = 0
    try:
        while True:
            item = game_queue.get()
            if item is None:
                break
            game_id, game_hash, pgn_text = item
            try:
                result = _process_game_with_engine(
                    engine, game_id, pgn_text, "lc0", evaluator_version,
                    min_time_control=min_time_control,
                    skip_bots=skip_bots,
                    humans_only=humans_only,
                )
                if result is None:
                    # Game filtered out — still mark in dedup so we don't rescan
                    result_queue.put({
                        "status": "filtered", "engine": "lc0",
                        "worker_id": worker_id,
                        "game_id": game_id, "game_hash": game_hash,
                    })
                    continue
                game_rec, moves, pmoves = result
                writer.write_game_data(game_rec, moves, pmoves)
                games_done += 1
                result_queue.put({
                    "status": "ok", "engine": "lc0",
                    "worker_id": worker_id,
                    "game_id": game_id, "game_hash": game_hash,
                    "num_moves": game_rec["ply_count"],
                    "num_possible_moves": len(pmoves),
                    "num_examples": writer.examples_written,
                })
            except Exception as e:
                result_queue.put({
                    "status": "error", "engine": "lc0",
                    "worker_id": worker_id,
                    "game_id": game_id, "game_hash": game_hash,
                    "error": str(e),
                })
    finally:
        writer.close()
        engine.quit()
        result_queue.put({
            "status": "worker_done", "engine": "lc0",
            "worker_id": worker_id, "games_done": games_done,
            "examples_written": writer.examples_written,
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
    lc0_min_nodes: int = 1,
    lc0_nodes_mult: float = 1.0,
    lc0_max_nodes: int = 0,
    num_lc0_workers: int = 8,
    lc0_version: str = "791556",
    use_direct_uci: bool = True,
    # General
    max_games: int = 0,
    # Dataset config
    max_possible: int = 220,
    shard_size: int = 500_000,
    val_pct: int = 10,
    test_pct: int = 10,
    with_phase: bool = False,
    checkpoint_interval: int = 100,
    min_time_control: int = 240,
    skip_bots: bool = True,
    humans_only: bool = False,
):
    """
    Main entry point. Reads PGN, deduplicates, fans out to LC0 worker pools,
    writes NPZ shards directly (no intermediate parquet).
    """
    if not lc0_path or not weights_path or num_lc0_workers <= 0:
        print("ERROR: Must specify --lc0, --weights, and --lc0-workers > 0.")
        sys.exit(1)

    dedup = GameDeduplicator(db_path)
    dedup.connect()

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Checkpoint for crash resume
    checkpoint_path = Path(output_dir) / 'checkpoint.json'
    resume_offset = 0
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as cf:
                ckpt = json.load(cf)
            resume_offset = ckpt.get('pgn_offset', 0)
            print(f"[MAIN] Resuming from checkpoint: offset={resume_offset}, "
                  f"previously dispatched={ckpt.get('games_dispatched', '?')}")
        except Exception as e:
            print(f"[MAIN] Warning: failed to read checkpoint: {e}")
            resume_offset = 0

    # Create output directories
    dataset_output = Path(output_dir)
    for split in ('train', 'val', 'test'):
        (dataset_output / split).mkdir(parents=True, exist_ok=True)

    # Clean up any .tmp shard files left by a previous crashed run.
    # These are incomplete atomic writes that never got renamed.
    tmp_cleaned = 0
    for split in ('train', 'val', 'test'):
        for tmp_file in (dataset_output / split).glob('*.tmp'):
            tmp_file.unlink()
            tmp_cleaned += 1
    if tmp_cleaned:
        print(f"[MAIN] Cleaned up {tmp_cleaned} leftover .tmp file(s) from previous crash")

    game_queue: Queue = Queue(maxsize=num_lc0_workers * 4)
    result_queue: Queue = Queue()

    all_workers: List[Process] = []

    # Spawn LC0 workers
    for wid in range(num_lc0_workers):
        p = Process(
            target=lc0_worker,
            args=(
                wid, game_queue, result_queue,
                lc0_path, weights_path, backend,
                lc0_batch_size, lc0_min_nodes, lc0_max_nodes, lc0_nodes_mult,
                min_time_control, skip_bots, humans_only,
                str(dataset_output), lc0_version, run_timestamp,
                max_possible, shard_size, val_pct, test_pct, with_phase,
                use_direct_uci,
            ),
            daemon=True,
        )
        p.start()
        all_workers.append(p)
    print(f"[MAIN] Spawned {num_lc0_workers} LC0 workers -> {dataset_output}")
    print(f"[MAIN] Mode: ALL legal moves evaluated → direct NPZ shards")
    print(f"[MAIN] Shard size: {shard_size:,} examples per shard")

    # Read PGN and dispatch
    dispatched = 0
    skipped = 0
    game_number = 0
    t0 = time.time()

    print(f"[MAIN] Reading PGN: {pgn_path}")
    print(f"[MAIN] Max games: {'unlimited' if max_games <= 0 else max_games}")
    print(f"[MAIN] Output: {dataset_output} (train/val/test NPZ shards)")

    collector_state = {
        "lc0_completed": 0,
        "errors": 0,
        "workers_done": 0,
        "total_possible_moves": 0,
        "total_examples": 0,
    }
    collector_done = threading.Event()

    def result_collector():
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

            if msg["status"] == "filtered":
                thread_dedup.mark_processed(msg["game_hash"], msg["game_id"], msg["engine"])
                collector_state["skipped_filter"] = collector_state.get("skipped_filter", 0) + 1
                continue
            if msg["status"] == "worker_done":
                collector_state["workers_done"] += 1
                print(
                    f"[LC0] Worker {msg['worker_id']} finished: "
                    f"{msg['games_done']} games, "
                    f"{msg['examples_written']} examples written"
                )
                if collector_state["workers_done"] >= num_lc0_workers:
                    thread_dedup.close()
                    break
            elif msg["status"] == "ok":
                thread_dedup.mark_processed(msg["game_hash"], msg["game_id"], msg["engine"])
                collector_state["total_possible_moves"] += msg.get("num_possible_moves", 0)
                collector_state["total_examples"] = max(
                    collector_state["total_examples"],
                    msg.get("num_examples", 0),
                )
                collector_state["lc0_completed"] += 1
                total_done = collector_state["lc0_completed"]
                if total_done % 20 == 0:
                    elapsed = time.time() - t0
                    rate = total_done / elapsed if elapsed > 0 else 0
                    print(
                        f"[MAIN] Completed: {total_done}/{dispatched} | "
                        f"PossibleMoves: {collector_state['total_possible_moves']:,} | "
                        f"{elapsed:.1f}s ({rate:.2f} g/s)"
                    )
            elif msg["status"] == "error":
                collector_state["errors"] += 1
                print(f"[LC0] Error: {msg.get('error', 'unknown')}")

    collector_thread = threading.Thread(target=result_collector, daemon=True)
    collector_thread.start()

    def _save_checkpoint(offset, dispatched_count):
        tmp = checkpoint_path.with_suffix('.json.tmp')
        with open(tmp, 'w') as cf:
            json.dump({'pgn_offset': offset, 'games_dispatched': dispatched_count,
                       'timestamp': time.time()}, cf)
        os.replace(tmp, checkpoint_path)

    with open(pgn_path, "r", errors="replace") as f:
        if resume_offset > 0:
            f.seek(resume_offset)
            print(f"[MAIN] Seeked to byte offset {resume_offset}")
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

            if dedup.is_duplicate(game_hash):
                skipped += 1
                continue

            game_queue.put((game_id, game_hash, pgn_text))
            dispatched += 1

            if dispatched % checkpoint_interval == 0:
                _save_checkpoint(f.tell(), dispatched)

            if game_number % 100 == 0:
                elapsed = time.time() - t0
                total_done = collector_state["lc0_completed"]
                print(
                    f"[MAIN] Scanned {game_number} games | "
                    f"Dispatched: {dispatched} | "
                    f"Completed: {total_done} | "
                    f"Skipped: {skipped} | "
                    f"{elapsed:.1f}s"
                )

    for _ in range(num_lc0_workers):
        game_queue.put(None)

    print(f"\n[MAIN] All {dispatched} games dispatched. Waiting for workers to finish...")

    collector_done.set()
    collector_thread.join(timeout=3600)

    for p in all_workers:
        p.join(timeout=15)

    lc0_completed = collector_state["lc0_completed"]
    errors = collector_state["errors"]

    dedup.close()
    elapsed = time.time() - t0

    filtered = collector_state.get("skipped_filter", 0)
    rate = lc0_completed / elapsed if elapsed > 0 else 0

    # Write dataset config
    config = {
        'format': 'direct_dataset_v1',
        'max_possible': max_possible,
        'shard_size': shard_size,
        'val_pct': val_pct,
        'test_pct': test_pct,
        'with_phase': with_phase,
        'engine': 'lc0',
        'engine_version': lc0_version,
        'total_games': lc0_completed,
        'total_errors': errors,
        'wall_time_seconds': elapsed,
    }
    config_path = dataset_output / 'dataset_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Count shards
    train_shards = len(list((dataset_output / 'train').glob('*.npz')))
    val_shards = len(list((dataset_output / 'val').glob('*.npz')))
    test_shards = len(list((dataset_output / 'test').glob('*.npz')))

    print(f"\n{'='*60}")
    print(f"  DIRECT DATASET PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  PGN games scanned:    {game_number}")
    print(f"  Games dispatched:     {dispatched}")
    print(f"  LC0 completed:        {lc0_completed}")
    print(f"  Total possible moves: {collector_state['total_possible_moves']:,}")
    print(f"  Errors:               {errors}")
    print(f"  Wall time:            {elapsed:.1f}s ({rate:.2f} g/s)")
    print(f"  Output:               {dataset_output}")
    print(f"  Train shards:         {train_shards}")
    print(f"  Val shards:           {val_shards}")
    print(f"  Test shards:          {test_shards}")
    print(f"  Config:               {config_path}")
    print(f"{'='*60}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Direct dataset processor — PGN → NPZ shards (no parquet)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # LC0 only, 4 workers, direct to NPZ:
  python parallel_dataset_processor.py games.pgn --lc0 lc0.exe --weights 791556.pb.gz --lc0-workers 4

  # Custom shard size and split ratios:
  python parallel_dataset_processor.py games.pgn \\
      --lc0 lc0.exe --weights 791556.pb.gz --lc0-workers 4 \\
      --shard-size 250000 --val-pct 10 --test-pct 5

  # With game phase labels:
  python parallel_dataset_processor.py games.pgn \\
      --lc0 lc0.exe --weights 791556.pb.gz --with-phase
""",
    )

    parser.add_argument("pgn_file", help="Path to PGN file")
    parser.add_argument("--db", default="chessv3.db", help="SQLite dedup database")
    parser.add_argument("--output", default="dataset_direct", help="Output directory for NPZ shards")
    parser.add_argument("--max-games", type=int, default=0, help="Max games (0=all)")

    # LC0
    lc0 = parser.add_argument_group("LC0")
    lc0.add_argument("--lc0", default="", help="Path to lc0 executable")
    lc0.add_argument("--weights", default="", help="LC0 weights file")
    lc0.add_argument("--backend", default="cuda-fp16", help="LC0 backend")
    lc0.add_argument("--lc0-batch", type=int, default=256,
                     help="LC0 minibatch size")
    lc0.add_argument("--lc0-min-nodes", type=int, default=1, help="Minimum nodes per position")
    lc0.add_argument("--lc0-nodes-mult", type=float, default=1.0, help="Nodes multiplier: nodes = max(min_nodes, n_legal * mult)")
    lc0.add_argument("--lc0-max-nodes", type=int, default=0, help="Hard cap on nodes per position (0=no cap)")
    lc0.add_argument("--lc0-workers", type=int, default=8, help="Number of LC0 workers")
    lc0.add_argument("--lc0-version", default="791556", help="LC0 version tag")
    lc0.add_argument("--use-direct-uci", action="store_true", default=True,
                     help="Use direct UCI subprocess (default, faster)")
    lc0.add_argument("--no-direct-uci", dest="use_direct_uci", action="store_false",
                     help="Use python-chess wrapper instead of direct UCI")

    # Dataset
    ds = parser.add_argument_group("Dataset")
    ds.add_argument("--max-possible", type=int, default=220,
                    help="Max candidate moves per position")
    ds.add_argument("--shard-size", type=int, default=500_000,
                    help="Examples per NPZ shard file")
    ds.add_argument("--val-pct", type=int, default=10,
                    help="Validation split percentage")
    ds.add_argument("--test-pct", type=int, default=10,
                    help="Test split percentage")
    ds.add_argument("--with-phase", action="store_true",
                    help="Include game_phase labels (opening/middle/endgame)")
    ds.add_argument("--checkpoint-interval", type=int, default=100,
                    help="Save PGN checkpoint every N games dispatched")

    # Filtering
    filt = parser.add_argument_group("Filtering")
    filt.add_argument("--min-time-control", type=int, default=240,
                      help="Min effective time control (start + 30*increment). Default 240 = 180+2")
    filt.add_argument("--skip-bots", action="store_true", default=True,
                      help="Skip games where both players are bots (default: on)")
    filt.add_argument("--no-skip-bots", dest="skip_bots", action="store_false",
                      help="Include bot-vs-bot games")
    filt.add_argument("--humans-only", action="store_true", default=True,
                      help="Only record moves where side-to-move is human (default: on)")
    filt.add_argument("--no-humans-only", dest="humans_only", action="store_false",
                      help="Record all moves including bot moves")

    args = parser.parse_args()

    run_parallel(
        pgn_path=args.pgn_file,
        db_path=args.db,
        output_dir=args.output,
        lc0_path=args.lc0,
        weights_path=args.weights,
        backend=args.backend,
        lc0_batch_size=args.lc0_batch,
        lc0_min_nodes=args.lc0_min_nodes,
        lc0_nodes_mult=args.lc0_nodes_mult,
        lc0_max_nodes=args.lc0_max_nodes,
        num_lc0_workers=args.lc0_workers,
        lc0_version=args.lc0_version,
        use_direct_uci=args.use_direct_uci,
        max_games=args.max_games,
        max_possible=args.max_possible,
        shard_size=args.shard_size,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        with_phase=args.with_phase,
        checkpoint_interval=args.checkpoint_interval,
        min_time_control=args.min_time_control,
        skip_bots=args.skip_bots,
        humans_only=args.humans_only,
    )


if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows multiprocessing
    main()
