"""
LC0 Game Processor with Parquet — rewritten for true game-level parallelism.

KEY CHANGE: The original sequential loop (one game at a time) meant that
adding more engines per worker had zero benefit — the bottleneck was the
serial game loop, not per-position throughput. This version dispatches
games to worker processes via multiprocessing, each with its own engine.

Architecture:
    Main process:
        - Reads PGN sequentially (not picklable)
        - Deduplicates via SQLite (GameDeduplicator)
        - Sends game PGN strings to worker pool
        - Collects results, marks dedup

    Worker processes (N of them):
        - Each creates its own LC0 engine
        - Processes one game at a time (all positions)
        - Returns dicts of game/move/possible_move records
        - Writes to its own parquet partition
"""

import os
import io
import time
import hashlib
import sqlite3
import multiprocessing as mp
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import asdict

import chess
import chess.pgn

from parquet_writer import (
    ParquetWriter,
    GameRecord,
    MoveRecord,
    PossibleMoveRecord,
)
from batch_evaluator import SyncBatchEvaluator


# ── Deduplication ────────────────────────────────────────────────────────────

class GameDeduplicator:
    """
    Deduplication using the existing games table.
    Checks games.game_hash + games.evaluated_by.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self):
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")

    def is_duplicate(self, game_hash: str) -> bool:
        if self._conn is None:
            self.connect()
        row = self._conn.execute(
            "SELECT 1 FROM games WHERE game_hash = ? AND evaluated_by = 'lc0' LIMIT 1",
            (game_hash,),
        ).fetchone()
        return row is not None

    def mark_processed(self, game_hash: str, game_id: str, engine: str = "lc0"):
        """No-op — game row is inserted by the worker with game_hash + evaluated_by."""
        pass

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


def compute_game_hash(pgn_text: str) -> str:
    """Deterministic hash of a game's PGN text."""
    return hashlib.sha256(pgn_text.strip().encode("utf-8")).hexdigest()[:16]


# ── Worker function (runs in child process) ──────────────────────────────────

def _lc0_worker(
    worker_id: int,
    game_queue: "mp.Queue",
    result_queue: "mp.Queue",
    lc0_path: str,
    weights_path: str,
    backend: str,
    batch_size: int,
    nodes: int,
    output_dir: str,
    multipv: int,
    evaluator_version: str,
):
    """
    Worker process: pulls PGN strings from game_queue, evaluates every
    position with its own LC0 engine, writes parquet, sends summary back.
    """
    # Each worker gets its own parquet partition
    worker_output = os.path.join(output_dir, f"worker_{worker_id:02d}")
    writer = ParquetWriter(worker_output, batch_size=5000, worker_id=worker_id)

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
            if item is None:  # poison pill
                break

            game_id, game_hash, pgn_text = item
            try:
                records = _process_single_game(
                    engine, game_id, pgn_text, multipv, evaluator_version,
                )
                # Write to parquet
                writer.write_game(records["game"])
                for mr in records["moves"]:
                    writer.write_move(mr)
                for pm in records["possible_moves"]:
                    writer.write_possible_move(pm)
                games_done += 1

                result_queue.put({
                    "status": "ok",
                    "worker_id": worker_id,
                    "game_id": game_id,
                    "game_hash": game_hash,
                    "num_moves": records["game"].num_moves,
                })
            except Exception as e:
                result_queue.put({
                    "status": "error",
                    "worker_id": worker_id,
                    "game_id": game_id,
                    "game_hash": game_hash,
                    "error": str(e),
                })
    finally:
        writer.close()
        engine.quit()
        result_queue.put({
            "status": "worker_done",
            "worker_id": worker_id,
            "games_done": games_done,
            "games_written": writer.games_written,
            "moves_written": writer.moves_written,
            "possible_moves_written": writer.possible_moves_written,
        })


def _process_single_game(
    engine: SyncBatchEvaluator,
    game_id: str,
    pgn_text: str,
    multipv: int,
    evaluator_version: str,
) -> Dict[str, Any]:
    """Evaluate all positions in a game, return record objects."""
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
        evaluated_by="lc0",
        evaluator_version=evaluator_version,
    )

    move_records: List[MoveRecord] = []
    possible_move_records: List[PossibleMoveRecord] = []

    node = game
    ply = 0
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        fen = board.fen()

        # Evaluate position BEFORE the move is made
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
        is_best = move_uci == top_move

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
            is_best_move=is_best,
            evaluated_by="lc0",
        )
        move_records.append(mr)

        # Possible moves from multipv
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
                    evaluated_by="lc0",
                )
                possible_move_records.append(pm)

        board.push(move)
        node = next_node
        ply += 1

    game_rec.num_moves = ply
    return {
        "game": game_rec,
        "moves": move_records,
        "possible_moves": possible_move_records,
    }


def _safe_int(val: str) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


# ── Main PGN reader + dispatcher ────────────────────────────────────────────

class LC0ParallelProcessor:
    """
    Parallel LC0 game processor.

    Reads PGN in the main process, deduplicates, dispatches to N worker
    processes each running their own LC0 engine, collects results.
    """

    def __init__(
        self,
        pgn_path: str,
        db_path: str,
        output_dir: str,
        lc0_path: str,
        weights_path: str,
        backend: str = "cuda-fp16",
        batch_size: int = 32,
        nodes: int = 1,
        num_workers: int = 2,
        max_games: int = 0,
        multipv: int = 3,
        evaluator_version: str = "791556",
    ):
        self.pgn_path = pgn_path
        self.db_path = db_path
        self.output_dir = output_dir
        self.lc0_path = lc0_path
        self.weights_path = weights_path
        self.backend = backend
        self.batch_size = batch_size
        self.nodes = nodes
        self.num_workers = num_workers
        self.max_games = max_games
        self.multipv = multipv
        self.evaluator_version = evaluator_version

    def process(self):
        """Run the full pipeline."""
        dedup = GameDeduplicator(self.db_path)
        dedup.connect()

        lc0_output = os.path.join(self.output_dir, "lc0", self.evaluator_version)
        os.makedirs(lc0_output, exist_ok=True)

        game_queue: mp.Queue = mp.Queue(maxsize=self.num_workers * 4)
        result_queue: mp.Queue = mp.Queue()

        # Spawn workers
        workers: List[Process] = []
        for wid in range(self.num_workers):
            p = Process(
                target=_lc0_worker,
                args=(
                    wid,
                    game_queue,
                    result_queue,
                    self.lc0_path,
                    self.weights_path,
                    self.backend,
                    self.batch_size,
                    self.nodes,
                    lc0_output,
                    self.multipv,
                    self.evaluator_version,
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)

        # Read PGN and dispatch
        dispatched = 0
        skipped = 0
        game_number = 0

        print(f"[LC0] Reading PGN: {self.pgn_path}")
        print(f"[LC0] Workers: {self.num_workers}")
        print(f"[LC0] Output: {lc0_output}")

        with open(self.pgn_path, "r", errors="replace") as f:
            while True:
                if self.max_games > 0 and dispatched >= self.max_games:
                    break

                game = chess.pgn.read_game(f)
                if game is None:
                    break

                game_number += 1
                pgn_text = str(game)
                game_hash = compute_game_hash(pgn_text)

                if dedup.is_duplicate(game_hash):
                    skipped += 1
                    continue

                game_id = f"game_{game_hash}"
                game_queue.put((game_id, game_hash, pgn_text))
                dispatched += 1

                if dispatched % 50 == 0:
                    print(
                        f"[LC0] Dispatched {dispatched} games "
                        f"(scanned {game_number}, skipped {skipped} dupes)"
                    )

        # Send poison pills
        for _ in workers:
            game_queue.put(None)

        # Collect results
        workers_done = 0
        completed = 0
        errors = 0

        while workers_done < self.num_workers:
            msg = result_queue.get()
            if msg["status"] == "worker_done":
                workers_done += 1
                print(
                    f"[LC0] Worker {msg['worker_id']} finished: "
                    f"{msg['games_done']} games, "
                    f"{msg['games_written']} game records, "
                    f"{msg['moves_written']} moves, "
                    f"{msg['possible_moves_written']} possible moves"
                )
            elif msg["status"] == "ok":
                dedup.mark_processed(msg["game_hash"], msg["game_id"], engine="lc0")
                completed += 1
                if completed % 10 == 0:
                    print(f"[LC0] Completed {completed}/{dispatched} games")
            elif msg["status"] == "error":
                errors += 1
                print(f"[LC0] Error on {msg['game_id']}: {msg['error']}")

        # Wait for processes to exit
        for p in workers:
            p.join(timeout=10)

        dedup.close()

        print(f"\n[LC0] === SUMMARY ===")
        print(f"  Games dispatched: {dispatched}")
        print(f"  Games completed:  {completed}")
        print(f"  Games errored:    {errors}")
        print(f"  Games skipped:    {skipped}")
        print(f"  Output dir:       {lc0_output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel LC0 game processor with parquet output")
    parser.add_argument("pgn_file", help="Path to PGN file")
    parser.add_argument("--db", default="games.db", help="SQLite dedup database path")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--lc0", default=r"C:\Users\micha\Personal\Coding\chess-clone\lc0\lc0.exe")
    parser.add_argument("--weights", default="791556.pb.gz", help="LC0 weights file")
    parser.add_argument("--backend", default="cuda-fp16", help="LC0 backend")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2, help="Number of LC0 worker processes")
    parser.add_argument("--max-games", type=int, default=0, help="Max games to process (0=all)")
    parser.add_argument("--multipv", type=int, default=3, help="Number of PVs per position")

    args = parser.parse_args()

    processor = LC0ParallelProcessor(
        pgn_path=args.pgn_file,
        db_path=args.db,
        output_dir=args.output,
        lc0_path=args.lc0,
        weights_path=args.weights,
        backend=args.backend,
        batch_size=args.batch_size,
        nodes=args.nodes,
        num_workers=args.workers,
        max_games=args.max_games,
        multipv=args.multipv,
    )
    processor.process()
