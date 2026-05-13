#!/usr/bin/env python3
"""
parallel_processor.py

Splits a PGN file into chunks and processes them in parallel with multiple
LC0 instances sharing the same GPU. Each worker gets its own LC0 engine.

Usage:
    python parallel_processor.py lichess_2025-01.pgn --workers 3 --weights weights/791556.pb.gz
"""

import argparse
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime
import chess.pgn


def split_pgn(pgn_path: str, num_chunks: int, output_dir: str) -> list:
    """Split a PGN file into N roughly equal chunk files."""
    print(f"Counting games in {pgn_path}...")
    game_count = 0
    game_offsets = []
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            offset = f.tell()
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_offsets.append(offset)
            game_count += 1
            if game_count % 10000 == 0:
                print(f"  Counted {game_count} games...")
    
    print(f"Total games: {game_count}")
    
    chunk_size = game_count // num_chunks
    remainder = game_count % num_chunks
    
    chunk_files = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as src:
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size + min(chunk_idx, remainder)
            end = start + chunk_size + (1 if chunk_idx < remainder else 0)
            
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:03d}.pgn")
            
            with open(chunk_path, 'w', encoding='utf-8') as dst:
                for game_idx in range(start, end):
                    src.seek(game_offsets[game_idx])
                    game = chess.pgn.read_game(src)
                    if game:
                        dst.write(str(game))
                        dst.write("\n\n")
            
            num_games = end - start
            chunk_files.append((chunk_path, num_games))
            print(f"  Chunk {chunk_idx}: {num_games} games -> {chunk_path}")
    
    return chunk_files


def run_worker(worker_id: int, chunk_path: str, db_path: str, output_dir: str,
               weights_path: str, backend: str, num_engines: int = 16, max_games: int = None):
    """Run a single worker processing one PGN chunk."""
    # Import here so each process gets its own modules
    from lc0_processor_with_parquet import LC0GameProcessorWithParquet
    
    # Each worker gets its own DB file to avoid SQLite locking
    worker_db = db_path.replace('.db', f'_worker{worker_id}.db')
    
    # Each worker writes to its own parquet subdirectory
    worker_output = os.path.join(output_dir, f"worker_{worker_id}")
    
    print(f"[Worker {worker_id}] Starting: {chunk_path}")
    print(f"[Worker {worker_id}] DB: {worker_db}")
    print(f"[Worker {worker_id}] Output: {worker_output}")
    
    try:
        processor = LC0GameProcessorWithParquet(
            db_path=worker_db,
            weights_path=weights_path,
            output_dir=worker_output,
            backend=backend,
            write_parquet=True,
            write_sqlite=True,
            num_engines=num_engines,
            verbose=True
        )
        
        results = processor.process_pgn_file(chunk_path, max_games=max_games)
        processor.close()
        
        print(f"[Worker {worker_id}] Done: {results['games_processed']} games, "
              f"{results['positions_evaluated']} positions")
        return results
        
    except Exception as e:
        print(f"[Worker {worker_id}] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def merge_parquet_outputs(output_dir: str, num_workers: int):
    """Merge worker parquet outputs into final directory."""
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    final_dir = os.path.join(output_dir, "merged")
    
    for table_name in ['games', 'moves', 'possible_moves']:
        all_files = []
        for w in range(num_workers):
            worker_dir = os.path.join(output_dir, f"worker_{w}")
            # Find parquet files recursively
            for root, dirs, files in os.walk(worker_dir):
                for f in files:
                    if f.endswith('.parquet') and table_name in f:
                        all_files.append(os.path.join(root, f))
        
        if not all_files:
            continue
        
        merge_dir = os.path.join(final_dir, table_name)
        Path(merge_dir).mkdir(parents=True, exist_ok=True)
        
        # Just copy/concat - don't load everything into memory
        for i, f in enumerate(all_files):
            dest = os.path.join(merge_dir, f"part_{i:05d}.parquet")
            import shutil
            shutil.copy2(f, dest)
        
        print(f"Merged {len(all_files)} {table_name} files -> {merge_dir}")


def main():
    parser = argparse.ArgumentParser(description='Parallel LC0 processor')
    parser.add_argument('pgn_file', help='PGN file to process')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of parallel workers (default: 3)')
    parser.add_argument('--db', default='chess.db', help='Base SQLite database path')
    parser.add_argument('--output-dir', default='output/parquet', help='Output directory')
    parser.add_argument('--weights', required=True, help='LC0 weights file')
    parser.add_argument('--backend', default='cuda-fp16', help='LC0 backend')
    parser.add_argument('--max-games', type=int, help='Max games per worker (for testing)')
    parser.add_argument('--chunks-dir', default='chunks', help='Temp directory for PGN chunks')
    parser.add_argument('--skip-split', action='store_true', help='Skip splitting (reuse existing chunks)')
    parser.add_argument('--num-engines', type=int, default=16,
                       help='Async LC0 engines per worker (default: 16)')
    parser.add_argument('--merge', action='store_true', help='Merge outputs after processing')
    
    args = parser.parse_args()
    
    print("="*70)
    print(f"Parallel LC0 Processor - {args.workers} workers")
    print("="*70)
    
    # Step 1: Split PGN
    if not args.skip_split:
        chunks = split_pgn(args.pgn_file, args.workers, args.chunks_dir)
    else:
        chunks = []
        for i in range(args.workers):
            chunk_path = os.path.join(args.chunks_dir, f"chunk_{i:03d}.pgn")
            if os.path.exists(chunk_path):
                chunks.append((chunk_path, 0))
        print(f"Reusing {len(chunks)} existing chunks")
    
    # Step 2: Launch workers
    print(f"\nLaunching {args.workers} workers...")
    start_time = datetime.now()
    
    processes = []
    for i, (chunk_path, num_games) in enumerate(chunks):
        p = mp.Process(
            target=run_worker,
            args=(i, chunk_path, args.db, args.output_dir,
                  args.weights, args.backend, args.num_engines, args.max_games)
        )
        p.start()
        processes.append(p)
        print(f"  Worker {i} started (PID {p.pid})")
    
    # Wait for all workers
    for p in processes:
        p.join()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print(f"All workers complete in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    
    # Step 3: Merge if requested
    if args.merge:
        print("\nMerging outputs...")
        merge_parquet_outputs(args.output_dir, args.workers)
    
    print("\nDone! Worker outputs in:")
    for i in range(args.workers):
        print(f"  {args.output_dir}/worker_{i}/")
    if args.merge:
        print(f"  Merged: {args.output_dir}/merged/")


if __name__ == '__main__':
    mp.set_start_method('spawn')  # Required on Windows
    main()
