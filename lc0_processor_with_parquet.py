"""
LC0 Game Processor with Parquet — rewritten for true game-level parallelism.

Uses parquet_schema.py schemas 1:1 matching SQLite.
Hashes player names, computes winner/loser/elo diffs, extracts full move data.
Reuses eval_after = eval_before of next position for zero extra engine calls.
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def hash_name(name: str) -> str:
    """SHA-256 hash a player name."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def compute_game_hash(pgn_text: str) -> str:
    """Deterministic hash of a game's PGN text."""
    return hashlib.sha256(pgn_text.strip().encode("utf-8")).hexdigest()[:16]


def _safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _parse_time_comment(comment: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse clock comment like [%clk 0:05:23] to extract time_remaining.
    Returns (time_remaining_seconds, None) — time_spent computed from deltas.
    """
    if not comment:
        return None, None
    import re
    match = re.search(r'\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]', comment)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
        return h * 3600 + m * 60 + s, None
    return None, None


def _extract_wdl(eval_result, perspective_white: bool) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Extract WDL percentages from eval result's multipv data.
    Returns (white_win_perc, black_win_perc, draw_perc) as 0.0-1.0.
    LC0 returns WDL from white's perspective in multipv entries if available.
    """
    if eval_result.multipv and len(eval_result.multipv) > 0:
        top = eval_result.multipv[0]
        w = top.get("wdl_w")
        d = top.get("wdl_d")
        l = top.get("wdl_l")
        if w is not None and d is not None and l is not None:
            total = w + d + l
            if total > 0:
                return w / total, l / total, d / total
    
    # Fallback: estimate from centipawn score using sigmoid
    score_cp = eval_result.score_cp
    if score_cp is not None:
        # Simple sigmoid approximation
        import math
        win_prob = 1.0 / (1.0 + math.exp(-score_cp / 200.0))
        # Rough draw estimate
        draw_prob = max(0.0, 0.3 - abs(score_cp) / 1000.0)
        white_win = win_prob * (1.0 - draw_prob)
        black_win = (1.0 - win_prob) * (1.0 - draw_prob)
        return white_win, black_win, draw_prob
    
    return None, None, None


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
        pass

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


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
            if item is None:  # poison pill
                break

            game_id, game_hash, pgn_text = item
            try:
                records = _process_single_game(
                    engine, game_id, game_hash, pgn_text, multipv, evaluator_version,
                )
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
                    "ply_count": records["game"].ply_count or 0,
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
    game_hash: str,
    pgn_text: str,
    multipv: int,
    evaluator_version: str,
) -> Dict[str, Any]:
    """Evaluate all positions in a game, return records matching parquet_schema.py."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Failed to parse PGN")

    headers = game.headers
    board = game.board()

    # ── Extract header fields ────────────────────────────────────────────
    white_raw = headers.get("White", "Unknown")
    black_raw = headers.get("Black", "Unknown")
    white_hashed = hash_name(white_raw)
    black_hashed = hash_name(black_raw)
    result = headers.get("Result", "*")

    white_elo = _safe_int(headers.get("WhiteElo", "0"))
    black_elo = _safe_int(headers.get("BlackElo", "0"))
    white_rating_diff = _safe_int(headers.get("WhiteRatingDiff")) if headers.get("WhiteRatingDiff") else None
    black_rating_diff = _safe_int(headers.get("BlackRatingDiff")) if headers.get("BlackRatingDiff") else None

    # Compute winner/loser
    winner = None
    loser = None
    winner_elo = None
    loser_elo = None
    winner_loser_elo_diff = None

    if result == "1-0":
        winner = white_hashed
        loser = black_hashed
        winner_elo = white_elo
        loser_elo = black_elo
        if white_elo and black_elo:
            winner_loser_elo_diff = white_elo - black_elo
    elif result == "0-1":
        winner = black_hashed
        loser = white_hashed
        winner_elo = black_elo
        loser_elo = white_elo
        if white_elo and black_elo:
            winner_loser_elo_diff = black_elo - white_elo

    # ── Build GameRecord ─────────────────────────────────────────────────
    game_rec = GameRecord(
        game_id=game_id,
        game_order=None,
        event=headers.get("Event", ""),
        site=headers.get("Site", ""),
        date_played=headers.get("Date", ""),
        round=headers.get("Round", ""),
        white=white_hashed,
        black=black_hashed,
        result=result,
        white_elo=white_elo,
        white_rating_diff=white_rating_diff,
        black_elo=black_elo,
        black_rating_diff=black_rating_diff,
        white_title=headers.get("WhiteTitle"),
        black_title=headers.get("BlackTitle"),
        winner=winner,
        winner_elo=winner_elo,
        loser=loser,
        loser_elo=loser_elo,
        winner_loser_elo_diff=winner_loser_elo_diff,
        eco=headers.get("ECO", ""),
        termination=headers.get("Termination"),
        time_control=headers.get("TimeControl", ""),
        utc_date=headers.get("UTCDate"),
        utc_time=headers.get("UTCTime"),
        variant=headers.get("Variant"),
        ply_count=0,  # updated at end
        game_hash=game_hash,
        evaluated_by="lc0",
        evaluator_version=evaluator_version,
        evaluated_at=None,
        pgn_text=pgn_text,
    )

    # ── Process moves ────────────────────────────────────────────────────
    move_records: List[MoveRecord] = []
    possible_move_records: List[PossibleMoveRecord] = []

    node = game
    ply = 0
    move_no = 1
    moves_list = []  # for game_to_position
    prev_time_remaining = None

    # Cache: eval_after of previous move = eval_before of current move
    cached_eval = None

    while node.variations:
        next_node = node.variation(0)
        actual_move = next_node.move
        fen_before = board.fen()
        color = "white" if board.turn == chess.WHITE else "black"
        player_hashed = white_hashed if board.turn == chess.WHITE else black_hashed
        move_no_pair = (ply // 2) + 1

        # ── Evaluate position BEFORE the move ────────────────────────────
        if cached_eval is not None:
            eval_before_result = cached_eval
        else:
            try:
                eval_before_result = engine.evaluate_position(board, multipv=multipv)
            except Exception:
                board.push(actual_move)
                node = next_node
                ply += 1
                if ply % 2 == 0:
                    move_no += 1
                cached_eval = None
                continue

        # Extract eval_before values
        eval_before_cp = eval_before_result.score_cp
        mate_count_before = eval_before_result.score_mate
        static_eval_before = eval_before_cp  # LC0 doesn't distinguish static vs search eval at nodes=0
        w_before, b_before, d_before = _extract_wdl(eval_before_result, board.turn == chess.WHITE)

        # ── Extract move details ─────────────────────────────────────────
        move_uci = actual_move.uci()
        try:
            move_san = board.san(actual_move)
        except Exception:
            move_san = move_uci

        from_square = chess.square_name(actual_move.from_square)
        to_square = chess.square_name(actual_move.to_square)
        piece_type = board.piece_at(actual_move.from_square)
        piece = piece_type.symbol() if piece_type else ""
        promotion = chess.piece_symbol(actual_move.promotion).upper() if actual_move.promotion else None

        # Time from comments
        comment = next_node.comment or ""
        time_remaining, _ = _parse_time_comment(comment)
        time_spent = None
        if prev_time_remaining is not None and time_remaining is not None:
            # Time spent = previous remaining - current remaining (same color)
            # This is approximate; proper calc needs per-color tracking
            pass

        moves_list.append(move_san)
        game_to_position = ' '.join(moves_list)

        # ── Make the move, evaluate AFTER ────────────────────────────────
        board.push(actual_move)
        fen_after = board.fen()

        try:
            eval_after_result = engine.evaluate_position(board, multipv=1)
            cached_eval = eval_after_result  # reuse as next move's eval_before
        except Exception:
            eval_after_result = None
            cached_eval = None

        eval_after_cp = None
        mate_count_after = None
        static_eval_after = None
        w_after, b_after, d_after = None, None, None
        if eval_after_result:
            eval_after_cp = eval_after_result.score_cp
            mate_count_after = eval_after_result.score_mate
            static_eval_after = eval_after_cp
            w_after, b_after, d_after = _extract_wdl(eval_after_result, board.turn == chess.WHITE)

        # ── Build MoveRecord ─────────────────────────────────────────────
        mr = MoveRecord(
            game_id=game_id,
            move_no=move_no,
            move_no_pair=move_no_pair,
            player=player_hashed,
            notation=move_san,
            move=move_uci,
            from_square=from_square,
            to_square=to_square,
            piece=piece,
            promotion=promotion,
            color=color,
            fen_before=fen_before,
            fen_after=fen_after,
            time_remaining=time_remaining,
            time_spent=time_spent,
            game_to_position=game_to_position,
            white_win_perc_before=w_before,
            black_win_perc_before=b_before,
            draw_perc_before=d_before,
            white_win_perc_after=w_after,
            black_win_perc_after=b_after,
            draw_perc_after=d_after,
            static_eval_before=static_eval_before,
            static_eval_after=static_eval_after,
            eval_before=eval_before_cp,
            mate_count_before=mate_count_before,
            eval_after=eval_after_cp,
            mate_count_after=mate_count_after,
            evaluated_by="lc0",
            evaluator_version=evaluator_version,
        )
        move_records.append(mr)

        # ── Build PossibleMoveRecords from multipv ───────────────────────
        if eval_before_result.multipv:
            # We need to go back to the position before the move to extract
            # possible move details
            board_before = chess.Board(fen_before)
            
            for rank, pv_entry in enumerate(eval_before_result.multipv):
                pm_uci = pv_entry.get("move_uci", "")
                pm_san = pv_entry.get("move_san", "")

                # Parse the possible move for square/piece info
                pm_from = ""
                pm_to = ""
                pm_piece = ""
                pm_promotion = None
                pm_fen_after = ""
                
                try:
                    pm_move = chess.Move.from_uci(pm_uci)
                    pm_from = chess.square_name(pm_move.from_square)
                    pm_to = chess.square_name(pm_move.to_square)
                    pm_piece_obj = board_before.piece_at(pm_move.from_square)
                    pm_piece = pm_piece_obj.symbol() if pm_piece_obj else ""
                    pm_promotion = chess.piece_symbol(pm_move.promotion).upper() if pm_move.promotion else None
                    
                    # Compute fen_after for this possible move
                    board_copy = board_before.copy()
                    board_copy.push(pm_move)
                    pm_fen_after = board_copy.fen()
                except Exception:
                    pass

                pm_score_cp = pv_entry.get("score_cp")
                pm_score_mate = pv_entry.get("score_mate")
                pm_nodes = pv_entry.get("nodes", 0)

                # WDL for this possible move
                pm_w, pm_b, pm_d = None, None, None
                pm_wdl_w = pv_entry.get("wdl_w")
                pm_wdl_d = pv_entry.get("wdl_d")
                pm_wdl_l = pv_entry.get("wdl_l")
                if pm_wdl_w is not None and pm_wdl_d is not None and pm_wdl_l is not None:
                    total = pm_wdl_w + pm_wdl_d + pm_wdl_l
                    if total > 0:
                        pm_w = pm_wdl_w / total
                        pm_b = pm_wdl_l / total
                        pm_d = pm_wdl_d / total

                pm = PossibleMoveRecord(
                    game_id=game_id,
                    move_no=move_no,
                    move_no_pair=move_no_pair,
                    notation=pm_san,
                    move=pm_uci,
                    from_square=pm_from,
                    to_square=pm_to,
                    piece=pm_piece,
                    promotion=pm_promotion,
                    color=color,
                    fen_before=fen_before,
                    fen_after=pm_fen_after,
                    eval=pm_score_cp,
                    mate_count=pm_score_mate,
                    white_win_perc=pm_w,
                    black_win_perc=pm_b,
                    draw_perc=pm_d,
                    nodes=pm_nodes,
                    depth=pv_entry.get("depth"),
                    pv=None,
                    evaluated_by="lc0",
                    evaluator_version=evaluator_version,
                )
                possible_move_records.append(pm)

        node = next_node
        ply += 1
        prev_time_remaining = time_remaining
        if color == "black":
            move_no += 1

    game_rec.ply_count = ply
    return {
        "game": game_rec,
        "moves": move_records,
        "possible_moves": possible_move_records,
    }


# ── Main PGN reader + dispatcher ────────────────────────────────────────────

class LC0ParallelProcessor:
    """
    Parallel LC0 game processor.
    Reads PGN in the main process, deduplicates, dispatches to N workers.
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

        game_queue = mp.Queue(maxsize=self.num_workers * 4)
        result_queue = mp.Queue()

        # Start workers
        workers = []
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
        print(f"Started {self.num_workers} LC0 worker(s)")

        # Read PGN and dispatch games
        games_dispatched = 0
        games_skipped = 0
        games_completed = 0
        games_errored = 0
        workers_done = 0
        t0 = time.time()

        with open(self.pgn_path, "r", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                if self.max_games > 0 and games_dispatched >= self.max_games:
                    break

                pgn_text = str(game)
                game_hash = compute_game_hash(pgn_text)

                if dedup.is_duplicate(game_hash):
                    games_skipped += 1
                    continue

                game_id = f"game_{game_hash}"
                game_queue.put((game_id, game_hash, pgn_text))
                games_dispatched += 1

                if games_dispatched % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"Dispatched {games_dispatched} games ({games_skipped} skipped) in {elapsed:.1f}s")

                # Drain results without blocking
                while not result_queue.empty():
                    try:
                        res = result_queue.get_nowait()
                        if res["status"] == "ok":
                            games_completed += 1
                        elif res["status"] == "error":
                            games_errored += 1
                            print(f"  Error in game {res['game_id']}: {res['error']}")
                    except Exception:
                        break

        # Send poison pills
        for _ in workers:
            game_queue.put(None)

        # Collect remaining results
        while workers_done < self.num_workers:
            res = result_queue.get(timeout=600)
            if res["status"] == "worker_done":
                workers_done += 1
                print(f"Worker {res['worker_id']} done: {res['games_done']} games, "
                      f"{res['games_written']} written, "
                      f"{res['moves_written']} moves, "
                      f"{res['possible_moves_written']} possible_moves")
            elif res["status"] == "ok":
                games_completed += 1
            elif res["status"] == "error":
                games_errored += 1

        for p in workers:
            p.join(timeout=30)

        dedup.close()

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"Processing complete in {elapsed:.1f}s")
        print(f"  Dispatched: {games_dispatched}")
        print(f"  Completed:  {games_completed}")
        print(f"  Errors:     {games_errored}")
        print(f"  Skipped:    {games_skipped}")
        print(f"  Output:     {lc0_output}")
        print(f"{'='*60}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LC0 parallel game processor with Parquet output")
    parser.add_argument("pgn_file", help="Path to PGN file")
    parser.add_argument("--db", default="chessv2.db", help="SQLite database path")
    parser.add_argument("--output-dir", default="output/parquet", help="Parquet output directory")
    parser.add_argument("--lc0-path", default=r"C:\Users\micha\Personal\Coding\chess-clone\lc0\lc0.exe",
                        help="Path to LC0 binary")
    parser.add_argument("--weights", required=True, help="Path to LC0 weights file")
    parser.add_argument("--backend", default="cuda-fp16", help="LC0 backend")
    parser.add_argument("--batch-size", type=int, default=32, help="LC0 minibatch size")
    parser.add_argument("--nodes", type=int, default=1, help="LC0 nodes per position")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--max-games", type=int, default=0, help="Max games to process (0=all)")
    parser.add_argument("--multipv", type=int, default=3, help="Number of PVs per position")
    parser.add_argument("--evaluator-version", default="791556", help="Evaluator version string")

    args = parser.parse_args()

    processor = LC0ParallelProcessor(
        pgn_path=args.pgn_file,
        db_path=args.db,
        output_dir=args.output_dir,
        lc0_path=args.lc0_path,
        weights_path=args.weights,
        backend=args.backend,
        batch_size=args.batch_size,
        nodes=args.nodes,
        num_workers=args.workers,
        max_games=args.max_games,
        multipv=args.multipv,
        evaluator_version=args.evaluator_version,
    )
    processor.process()
