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
"""

Each game is sent to BOTH an LC0 worker AND a Stockfish worker.
Deduplication tracks engine type so a game can be processed by both.
"""

import os
import io
import sys
import time
import signal
import hashlib
import sqlite3
import argparse
import threading
import multiprocessing as mp
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional, List

import chess
import chess.pgn

from parquet_writer import (
    ParquetWriter,
    GameRecord,
    MoveRecord,
    PossibleMoveRecord,
)
from batch_evaluator import SyncBatchEvaluator
from stockfish_evaluator import StockfishEvaluator


# ── Deduplication ────────────────────────────────────────────────────────────

class GameDeduplicator:
    """
    SQLite-backed game deduplication.
    If pointed at an existing db, uses the games table.
    If db doesn't exist, creates a fresh one with a lightweight tracking table.
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


def _process_game_with_engine(engine, game_id, pgn_text, multipv, engine_name, engine_version):
    """
    Process a single game with any engine that has .evaluate_position(board, multipv).
    Returns (GameRecord, [MoveRecord], [PossibleMoveRecord]).
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Failed to parse PGN")

    headers = game.headers
    board = game.board()

    game_rec = GameRecord(
        game_id=game_id,
        event=headers.get("Event", ""),
        site=headers.get("Site", ""),
        date=headers.get("Date", ""),
        round=headers.get("Round", ""),
        white=headers.get("White", ""),
        black=headers.get("Black", ""),
        result=headers.get("Result", ""),
        white_elo=_safe_int(headers.get("WhiteElo", "0")),
        black_elo=_safe_int(headers.get("BlackElo", "0")),
        time_control=headers.get("TimeControl", ""),
        eco=headers.get("ECO", ""),
        opening=headers.get("Opening", ""),
        pgn_text=pgn_text,
        num_moves=0,
        evaluated_by=engine_name,
        evaluator_version=engine_version,
    )

    move_records = []
    possible_move_records = []
    node = game
    ply = 0

    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        fen = board.fen()

        try:
            eval_result = engine.evaluate_position(board, multipv=multipv)
        except Exception:
            board.push(move)
            node = next_node
            ply += 1
            continue

        move_uci = move.uci()
        try:
            move_san = board.san(move)
        except Exception:
            move_san = move_uci

        top_move = eval_result.best_move or ""
        top_san = eval_result.best_move_san or ""

        mr = MoveRecord(
            game_id=game_id,
            ply=ply,
            fen=fen,
            move_uci=move_uci,
            move_san=move_san,
            score_cp=eval_result.score_cp,
            score_mate=eval_result.score_mate,
            top_move_uci=top_move,
            top_move_san=top_san,
            top_score_cp=eval_result.score_cp,
            top_score_mate=eval_result.score_mate,
            is_best_move=(move_uci == top_move),
            evaluated_by=engine_name,
        )
        move_records.append(mr)

        if eval_result.multipv:
            for rank, pv_entry in enumerate(eval_result.multipv):
                pm = PossibleMoveRecord(
                    game_id=game_id,
                    ply=ply,
                    fen=fen,
                    move_uci=pv_entry.get("move_uci", ""),
                    move_san=pv_entry.get("move_san", ""),
                    score_cp=pv_entry.get("score_cp"),
                    score_mate=pv_entry.get("score_mate"),
                    rank=rank,
                    prior_probability=0.0,
                    visits=pv_entry.get("nodes", 0),
                    evaluated_by=engine_name,
                )
                possible_move_records.append(pm)

        board.push(move)
        node = next_node
        ply += 1

    game_rec.num_moves = ply
    return game_rec, move_records, possible_move_records


# ── LC0 Worker ───────────────────────────────────────────────────────────────

def lc0_worker(
    worker_id, game_queue, result_queue,
    lc0_path, weights_path, backend, batch_size, nodes,
    output_dir, multipv, evaluator_version,
):
    """LC0 worker process."""
    worker_output = os.path.join(output_dir, f"worker_{worker_id:02d}")
    writer = ParquetWriter(worker_output, worker_id=worker_id)

    engine = SyncBatchEvaluator(
        lc0_path=lc0_path,
        weights_path=weights_path,
        backend=backend,
        batch_size=batch_size,
        nodes=nodes,
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
                game_rec, moves, pmoves = _process_game_with_engine(
                    engine, game_id, pgn_text, multipv, "lc0", evaluator_version,
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
                    "num_moves": game_rec.num_moves,
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
            "games_written": writer.games_written,
            "moves_written": writer.moves_written,
            "possible_moves_written": writer.possible_moves_written,
        })


# ── Stockfish Worker ─────────────────────────────────────────────────────────

def stockfish_worker(
    worker_id, game_queue, result_queue,
    stockfish_path, depth, threads, hash_mb,
    output_dir, multipv,
):
    """Stockfish depth-14 worker process."""
    worker_output = os.path.join(output_dir, f"worker_{worker_id:02d}")
    writer = ParquetWriter(worker_output, worker_id=worker_id)

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
                    engine, game_id, pgn_text, multipv, "stockfish", sf_version,
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
                    "num_moves": game_rec.num_moves,
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
    multipv: int = 1,
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

    dedup = GameDeduplicator(db_path)
    dedup.connect()

    lc0_output = os.path.join(output_dir, "lc0", lc0_version)
    sf_output = os.path.join(output_dir, "stockfish_d14", "latest")

    # Single shared queue — each game goes to ONE engine, not both
    game_queue: Queue = Queue(maxsize=(num_lc0_workers + num_sf_workers) * 4)
    result_queue: Queue = Queue()

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
                ),
                daemon=True,
            )
            p.start()
            all_workers.append(p)
            total_worker_count += 1
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
                    sf_output, multipv,
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
    }
    collector_done = threading.Event()

    def result_collector():
        """Drain result_queue in background, write dedup entries immediately."""
        while True:
            try:
                msg = result_queue.get(timeout=5)
            except Exception:
                if collector_done.is_set():
                    break
                continue

            if msg["status"] == "worker_done":
                collector_state["workers_done"] += 1
                eng = msg["engine"]
                print(
                    f"[{eng.upper()}] Worker {msg['worker_id']} finished: "
                    f"{msg['games_done']} games, "
                    f"{msg['moves_written']} moves written"
                )
                if collector_state["workers_done"] >= total_worker_count:
                    break
            elif msg["status"] == "ok":
                dedup.mark_processed(msg["game_hash"], msg["game_id"], msg["engine"])
                if msg["engine"] == "lc0":
                    collector_state["lc0_completed"] += 1
                else:
                    collector_state["sf_completed"] += 1
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

    print(f"\n{'='*60}")
    print(f"  PARALLEL PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  PGN games scanned:    {game_number}")
    print(f"  Games dispatched:     {dispatched}")
    print(f"  LC0 completed:        {lc0_completed}")
    print(f"  Stockfish completed:  {sf_completed}")
    print(f"  Errors:               {errors}")
    print(f"  Wall time:            {elapsed:.1f}s")
    print(f"  LC0 output:           {lc0_output}")
    if use_sf:
        print(f"  Stockfish output:     {sf_output}")
    print(f"{'='*60}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel chess game processor — LC0 + Stockfish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # LC0 only, 4 workers:
  python parallel_processor.py games.pgn --lc0 lc0.exe --weights 791556.pb.gz --lc0-workers 4

  # Stockfish only, 8 workers at depth 14:
  python parallel_processor.py games.pgn --stockfish stockfish.exe --sf-workers 8

  # Both engines in parallel:
  python parallel_processor.py games.pgn \\
      --lc0 lc0.exe --weights 791556.pb.gz --lc0-workers 2 \\
      --stockfish stockfish.exe --sf-workers 4

  # Limit to 100 games:
  python parallel_processor.py games.pgn --max-games 100 \\
      --lc0 lc0.exe --weights 791556.pb.gz \\
      --stockfish stockfish.exe
""",
    )

    parser.add_argument("pgn_file", help="Path to PGN file")
    parser.add_argument("--db", default="chessv3.db", help="SQLite dedup database (created fresh if doesn't exist)")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--max-games", type=int, default=0, help="Max games (0=all)")
    parser.add_argument("--multipv", type=int, default=1, help="Multi-PV lines")

    # LC0
    lc0 = parser.add_argument_group("LC0")
    lc0.add_argument("--lc0", default="", help="Path to lc0 executable")
    lc0.add_argument("--weights", default="", help="LC0 weights file")
    lc0.add_argument("--backend", default="cuda-fp16", help="LC0 backend")
    lc0.add_argument("--lc0-batch", type=int, default=32, help="LC0 minibatch size")
    lc0.add_argument("--lc0-nodes", type=int, default=1, help="LC0 nodes per position")
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
        num_lc0_workers=args.lc0_workers,
        lc0_version=args.lc0_version,
        stockfish_path=args.stockfish,
        sf_depth=args.sf_depth,
        sf_threads=args.sf_threads,
        sf_hash_mb=args.sf_hash,
        num_sf_workers=args.sf_workers,
        max_games=args.max_games,
        multipv=args.multipv,
    )


if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows multiprocessing
    main()