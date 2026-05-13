#!/usr/bin/env python3
"""
parallel_processor.py

Splits a PGN file into chunks and processes them in parallel with multiple
LC0 instances sharing the same GPU. Each worker gets its own LC0 engine(s).

Architecture:
- Main process: scans PGN, builds chunks, launches workers + writer
- Worker processes: evaluate positions with LC0, send results to writer queue
- Writer process: single process owns the SQLite DB, serializes all writes

For large files (>2GB), workers seek directly into the original file — no temp copies.

Usage:
    python parallel_processor.py lichess_2025-01.pgn --workers 3 --num-engines 3 --weights weights/791556.pb.gz
"""

import argparse
import multiprocessing as mp
import os
import sqlite3
import json
import sys
from pathlib import Path
from datetime import datetime
from queue import Empty


# ── Writer Process ──────────────────────────────────────────────────────

WRITER_CMD_GAME = 'game'
WRITER_CMD_DEDUP = 'dedup'
WRITER_CMD_STOP = 'stop'
WRITER_CMD_IS_PROCESSED = 'is_processed'


def db_writer_process(db_path: str, write_queue: mp.Queue, response_queues: dict):
    """
    Dedicated writer process. Owns the single SQLite connection.
    All DB writes and dedup checks go through here.
    """
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")

    # Ensure tables exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_games (
            game_hash TEXT PRIMARY KEY,
            file_path TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            game_order INTEGER,
            event TEXT,
            site TEXT,
            date_played TEXT,
            round TEXT,
            white TEXT,
            black TEXT,
            result TEXT,
            white_elo INTEGER,
            white_rating_diff INTEGER,
            black_elo INTEGER,
            black_rating_diff INTEGER,
            white_title TEXT,
            black_title TEXT,
            winner TEXT,
            winner_elo INTEGER,
            loser TEXT,
            loser_elo INTEGER,
            winner_loser_elo_diff INTEGER,
            eco TEXT,
            termination TEXT,
            time_control TEXT,
            utc_date TEXT,
            utc_time TEXT,
            variant TEXT,
            ply_count INTEGER,
            game_hash TEXT,
            evaluated_by TEXT,
            evaluator_version TEXT,
            evaluated_at TEXT
        )
    """)
    conn.commit()

    batch_count = 0
    print("[Writer] Ready")

    while True:
        try:
            msg = write_queue.get(timeout=5)
        except Empty:
            continue

        cmd = msg[0]

        if cmd == WRITER_CMD_STOP:
            conn.commit()
            conn.close()
            print("[Writer] Stopped")
            return

        elif cmd == WRITER_CMD_IS_PROCESSED:
            worker_id, game_hash = msg[1], msg[2]
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM processed_games WHERE game_hash = ? LIMIT 1",
                [game_hash]
            )
            found = cursor.fetchone() is not None
            response_queues[worker_id].put(found)

        elif cmd == WRITER_CMD_GAME:
            header_data = msg[1]
            cols = ', '.join(header_data.keys())
            placeholders = ', '.join(f':{k}' for k in header_data.keys())
            try:
                conn.execute(f"INSERT OR REPLACE INTO games ({cols}) VALUES ({placeholders})", header_data)
            except Exception as e:
                print(f"[Writer] Error inserting game: {e}")

        elif cmd == WRITER_CMD_DEDUP:
            game_hash, file_path, metadata = msg[1], msg[2], msg[3]
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO processed_games (game_hash, file_path, metadata) VALUES (?, ?, ?)",
                    [game_hash, file_path, json.dumps(metadata) if metadata else None]
                )
            except Exception as e:
                print(f"[Writer] Error marking processed: {e}")

            batch_count += 1
            if batch_count % 50 == 0:
                conn.commit()


# ── PGN Scanning ────────────────────────────────────────────────────────

def scan_game_boundaries(pgn_path: str, max_games: int = None) -> list:
    """
    Fast scan for game start byte offsets.
    Looks for '[' after a blank line. ~50-100x faster than parsing.
    """
    print(f"Scanning {pgn_path} for game boundaries...")

    game_starts = []
    target = max_games or float('inf')

    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        prev_blank = True

        while len(game_starts) < target:
            offset = f.tell()
            line = f.readline()
            if not line:
                break

            if prev_blank and line.startswith('['):
                game_starts.append(offset)
                if len(game_starts) % 50000 == 0:
                    print(f"  Found {len(game_starts):,} games...")
                prev_blank = False
            elif line.strip() == '':
                prev_blank = True
            else:
                prev_blank = False

    print(f"Found {len(game_starts):,} games")
    return game_starts


def build_chunks(pgn_path: str, game_starts: list, num_chunks: int) -> list:
    """
    Divide game offsets into chunk specs.
    Returns list of (pgn_path, start_byte, end_byte, num_games).
    """
    total = len(game_starts)
    if total == 0:
        return []

    file_size = os.path.getsize(pgn_path)
    chunk_size = total // num_chunks
    remainder = total % num_chunks

    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size + min(i, remainder)
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)

        if start_idx >= total:
            break

        start_byte = game_starts[start_idx]
        end_byte = game_starts[end_idx] if end_idx < total else file_size
        num_games = end_idx - start_idx

        chunks.append((pgn_path, start_byte, end_byte, num_games))
        print(f"  Chunk {i}: {num_games:,} games @ byte {start_byte:,}-{end_byte:,}")

    return chunks


# ── Worker Process ──────────────────────────────────────────────────────

def run_worker(worker_id: int, pgn_path: str, start_byte: int, end_byte: int,
               output_dir: str, weights_path: str, engine_path: str,
               backend: str, num_engines: int,
               write_queue: mp.Queue, response_queue: mp.Queue):
    """Run a single worker reading a byte range from the PGN file."""
    import chess.pgn
    from lc0_processor_with_parquet import LC0GameProcessorWithParquet

    worker_output = os.path.join(output_dir, f"worker_{worker_id}")

    print(f"[Worker {worker_id}] Starting: bytes {start_byte:,}-{end_byte:,}")
    print(f"[Worker {worker_id}] Output: {worker_output}")

    try:
        # Create processor with SQLite disabled — writer process handles DB
        processor = LC0GameProcessorWithParquet(
            db_path=None,  # No direct DB access
            weights_path=weights_path,
            output_dir=worker_output,
            engine_path=engine_path,
            backend=backend,
            write_parquet=True,
            write_sqlite=False,  # Writer process handles this
            num_engines=num_engines,
            verbose=False
        )

        games_processed = 0
        games_skipped = 0
        positions_evaluated = 0
        possible_moves_written = 0
        game_order = 0
        start_time = datetime.now()

        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(start_byte)

            while f.tell() < end_byte:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                game_order += 1
                game_id = processor._game_hash(game)

                # Check dedup via writer process
                write_queue.put((WRITER_CMD_IS_PROCESSED, worker_id, game_id))
                is_processed = response_queue.get(timeout=30)

                if is_processed:
                    games_skipped += 1
                    continue

                try:
                    result = processor._process_game(
                        conn=None,
                        game=game,
                        game_id=game_id,
                        game_order=game_order,
                    )

                    positions_evaluated += result['positions']
                    possible_moves_written += result['possible_moves']
                    games_processed += 1

                    # Send game header to writer
                    game_data = processor._build_game_record(game, game_id, game_order)
                    header_cols = ['game_id', 'game_order', 'event', 'site', 'date_played', 'round',
                                  'white', 'black', 'result', 'white_elo', 'white_rating_diff',
                                  'black_elo', 'black_rating_diff', 'white_title', 'black_title',
                                  'winner', 'winner_elo', 'loser', 'loser_elo', 'winner_loser_elo_diff',
                                  'eco', 'termination', 'time_control', 'utc_date', 'utc_time',
                                  'variant', 'ply_count', 'game_hash', 'evaluated_by',
                                  'evaluator_version', 'evaluated_at']
                    header_data = {k: game_data[k] for k in header_cols if k in game_data}
                    write_queue.put((WRITER_CMD_GAME, header_data))

                    # Mark dedup
                    write_queue.put((WRITER_CMD_DEDUP, game_id, pgn_path,
                                    {'engine': 'lc0', 'positions': result['positions']}))

                    if games_processed % 10 == 0:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        rate = positions_evaluated / elapsed if elapsed > 0 else 0
                        gps = games_processed / elapsed if elapsed > 0 else 0
                        print(f"  [W{worker_id}] {games_processed:,} games | "
                              f"{positions_evaluated:,} pos | {rate:.0f} pos/s | {gps:.2f} games/s")

                except Exception as e:
                    print(f"  [W{worker_id}] Error on game {game_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Close parquet writers
        for writer in processor.parquet_writers.values():
            writer.close()
        processor.evaluator.close()

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[Worker {worker_id}] Done: {games_processed:,} games, "
              f"{positions_evaluated:,} positions in {elapsed/60:.1f}min")
        return {'games': games_processed, 'positions': positions_evaluated}

    except Exception as e:
        print(f"[Worker {worker_id}] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


# ── Merge ───────────────────────────────────────────────────────────────

def merge_parquet_outputs(output_dir: str, num_workers: int):
    """Merge worker parquet outputs into final directory."""
    import shutil

    final_dir = os.path.join(output_dir, "merged")

    for table_name in ['games', 'moves', 'possible_moves']:
        all_files = []
        for w in range(num_workers):
            worker_dir = os.path.join(output_dir, f"worker_{w}")
            for root, dirs, files in os.walk(worker_dir):
                for f in files:
                    if f.endswith('.parquet') and table_name in f:
                        all_files.append(os.path.join(root, f))

        if not all_files:
            continue

        merge_dir = os.path.join(final_dir, table_name)
        Path(merge_dir).mkdir(parents=True, exist_ok=True)

        for i, f in enumerate(all_files):
            dest = os.path.join(merge_dir, f"part_{i:05d}.parquet")
            shutil.copy2(f, dest)

        print(f"Merged {len(all_files)} {table_name} files -> {merge_dir}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Parallel LC0 processor')
    parser.add_argument('pgn_file', help='PGN file to process')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of parallel workers (default: 3)')
    parser.add_argument('--db', default='chess.db', help='SQLite database path')
    parser.add_argument('--output-dir', default='output/parquet', help='Output directory')
    parser.add_argument('--weights', required=True, help='LC0 weights file')
    parser.add_argument('--engine-path', default=None, help='Path to lc0.exe')
    parser.add_argument('--backend', default='cuda-fp16', help='LC0 backend')
    parser.add_argument('--num-engines', type=int, default=16,
                       help='Async LC0 engines per worker (default: 16)')
    parser.add_argument('--max-games', type=int, help='Max total games to process')
    parser.add_argument('--merge', action='store_true', help='Merge outputs after processing')

    args = parser.parse_args()

    print("="*70)
    print(f"Parallel LC0 Processor")
    print(f"  Workers: {args.workers}")
    print(f"  Engines/worker: {args.num_engines}")
    print(f"  Total LC0 instances: {args.workers * args.num_engines}")
    print(f"  DB: {args.db}")
    print("="*70)

    # Step 1: Scan for game boundaries
    game_starts = scan_game_boundaries(args.pgn_file, args.max_games)

    if not game_starts:
        print("No games found.")
        return

    # Step 2: Build chunk specs (no temp files)
    chunks = build_chunks(args.pgn_file, game_starts, args.workers)

    # Step 3: Start writer process
    write_queue = mp.Queue()
    response_queues = {i: mp.Queue() for i in range(len(chunks))}

    writer = mp.Process(
        target=db_writer_process,
        args=(args.db, write_queue, response_queues)
    )
    writer.start()
    print(f"  Writer started (PID {writer.pid})")

    # Step 4: Launch workers
    print(f"\nLaunching {len(chunks)} workers...")
    start_time = datetime.now()

    processes = []
    for i, (pgn_path, start_byte, end_byte, num_games) in enumerate(chunks):
        p = mp.Process(
            target=run_worker,
            args=(i, pgn_path, start_byte, end_byte,
                  args.output_dir, args.weights, args.engine_path,
                  args.backend, args.num_engines,
                  write_queue, response_queues[i])
        )
        p.start()
        processes.append(p)
        print(f"  Worker {i} started (PID {p.pid}) — {num_games:,} games")

    # Wait for workers
    for p in processes:
        p.join()

    # Stop writer
    write_queue.put((WRITER_CMD_STOP,))
    writer.join(timeout=30)

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*70}")
    print(f"All workers complete in {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"{'='*70}")

    # Step 5: Merge if requested
    if args.merge:
        print("\nMerging outputs...")
        merge_parquet_outputs(args.output_dir, len(chunks))

    print("\nDone! Outputs:")
    print(f"  SQLite: {args.db}")
    for i in range(len(chunks)):
        print(f"  Parquet: {args.output_dir}/worker_{i}/")
    if args.merge:
        print(f"  Merged: {args.output_dir}/merged/")


if __name__ == '__main__':
    mp.set_start_method('spawn')  # Required on Windows
    main()
