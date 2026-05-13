#!/usr/bin/env python3
"""
lc0_processor_with_parquet.py

UPDATED VERSION - Writes to BOTH SQLite and Parquet with engine tracking

This version properly:
1. Annotates ALL tables (games, actual_moves, possible_move_evals) with evaluated_by
2. Writes to Parquet files partitioned by engine type
3. Maintains same schema in both SQLite and Parquet
4. Allows filtering by engine at training time from either source

Key improvements over lc0_processor.py:
- Added ParquetWriter integration
- Possible moves written to Parquet with engine annotations
- Partitioned output: output/lc0/ or output/stockfish/
- Schema validation to ensure Parquet matches SQLite
"""

import sqlite3
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import chess.pgn

from lc0_evaluator import LC0Evaluator
from plane_codec import board_to_planes, pack_planes
from deduplication_helpers import GameDeduplicator

# Import Parquet writer if available
try:
    from parquet_writer import ParquetWriter, GameRecord, MoveRecord, PossibleMoveRecord
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("Warning: parquet_writer not available. Install pyarrow: pip install pyarrow")


class LC0GameProcessorWithParquet:
    """
    LC0 processor that writes to BOTH SQLite and Parquet
    
    Output structure:
    - SQLite: chess.db (with evaluated_by columns)
    - Parquet: output/lc0/games/, output/lc0/moves/, output/lc0/possible_moves/
    
    All records include:
    - evaluated_by: 'lc0' or 'stockfish'
    - evaluator_version: network hash or SF version
    - evaluated_at: timestamp
    """
    
    def __init__(self,
                 db_path: str,
                 weights_path: str,
                 output_dir: str = "output",
                 backend: str = "cuda-fp16",
                 write_parquet: bool = True,
                 write_sqlite: bool = True,
                 verbose: bool = True):
        """
        Initialize processor with dual output support.
        
        Args:
            db_path: Path to SQLite database
            weights_path: Path to LC0 network weights
            output_dir: Base output directory for Parquet files
            backend: LC0 backend
            write_parquet: Write to Parquet files
            write_sqlite: Write to SQLite database
            verbose: Print progress
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.write_parquet = write_parquet and PARQUET_AVAILABLE
        self.write_sqlite = write_sqlite
        self.verbose = verbose
        
        if write_parquet and not PARQUET_AVAILABLE:
            raise ImportError("Parquet writing requested but parquet_writer not available. "
                            "Install with: pip install pyarrow")
        
        # Initialize LC0 evaluator
        print(f"Initializing LC0 processor with {backend} backend...")
        self.evaluator = LC0Evaluator(
            weights_path=weights_path,
            backend=backend,
            threads=4,
            max_batch_size=256,
            verbose=False
        )
        
        # Initialize deduplicator
        self.deduplicator = GameDeduplicator(db_path)
        
        # Engine metadata for tracking
        self.engine_info = {
            'engine': 'lc0',
            'version': self.evaluator.network_info['weights_hash'],
            'backend': backend,
            'weights_file': Path(weights_path).name,
            'evaluated_at': datetime.now().isoformat(),
        }
        
        # Initialize Parquet writers
        self.parquet_writers = {}
        if self.write_parquet:
            # Create engine-specific output directory
            engine_dir = self.output_dir / 'lc0' / self.engine_info['version']
            engine_dir.mkdir(parents=True, exist_ok=True)
            
            self.parquet_writers = {
                'games': ParquetWriter(
                    output_dir=str(engine_dir / 'games'),
                    schema_type='games',
                    max_rows_per_file=10000
                ),
                'moves': ParquetWriter(
                    output_dir=str(engine_dir / 'moves'),
                    schema_type='moves',
                    max_rows_per_file=100000
                ),
                'possible_moves': ParquetWriter(
                    output_dir=str(engine_dir / 'possible_moves'),
                    schema_type='possible_moves',
                    max_rows_per_file=500000
                ),
            }
            print(f"Parquet output: {engine_dir}")
        
        print(f"Processor ready:")
        print(f"  Engine: lc0-{self.engine_info['version']}")
        print(f"  SQLite: {'enabled' if self.write_sqlite else 'disabled'} ({db_path})")
        print(f"  Parquet: {'enabled' if self.write_parquet else 'disabled'}")
    
    def _game_hash(self, game: chess.pgn.Game) -> str:
        """Generate hash for deduplication"""
        headers = game.headers
        moves = ''.join([move.uci() for move in game.mainline_moves()])
        content = f"{headers.get('White', '')}|{headers.get('Black', '')}|{headers.get('Date', '')}|{moves}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def process_pgn_file(self, pgn_path: str, max_games: Optional[int] = None) -> Dict:
        """Process PGN file and write to both SQLite and Parquet"""
        pgn_path = Path(pgn_path)
        if not pgn_path.exists():
            raise FileNotFoundError(f"PGN file not found: {pgn_path}")
        
        print(f"\n{'='*70}")
        print(f"Processing {pgn_path.name} with LC0")
        print(f"Engine: {self.engine_info['engine']} {self.engine_info['version']}")
        print(f"{'='*70}")
        
        # Open SQLite connection if needed
        conn = None
        if self.write_sqlite:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")
        
        games_processed = 0
        games_skipped = 0
        positions_evaluated = 0
        possible_moves_written = 0
        
        start_time = datetime.now()
        
        try:
            with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn:
                while True:
                    if max_games and games_processed >= max_games:
                        break
                    
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    
                    game_id = self._game_hash(game)
                    
                    # Check if already processed
                    if self.deduplicator.is_game_processed(game_id):
                        games_skipped += 1
                        if games_skipped % 100 == 0 and self.verbose:
                            print(f"  Skipped {games_skipped} duplicates...")
                        continue
                    
                    try:
                        # Process game
                        result = self._process_game(
                            conn=conn,
                            game=game,
                            game_id=game_id
                        )
                        
                        positions_evaluated += result['positions']
                        possible_moves_written += result['possible_moves']
                        games_processed += 1
                        
                        # Mark as processed
                        self.deduplicator.mark_game_processed(
                            game_id=game_id,
                            file_path=str(pgn_path),
                            metadata={
                                'engine': 'lc0',
                                'version': self.engine_info['version'],
                                'positions': result['positions'],
                                'possible_moves': result['possible_moves'],
                            }
                        )
                        
                        if games_processed % 10 == 0 and self.verbose:
                            elapsed = (datetime.now() - start_time).total_seconds()
                            rate = positions_evaluated / elapsed if elapsed > 0 else 0
                            print(f"  [{games_processed}] {positions_evaluated:,} positions, "
                                  f"{possible_moves_written:,} possible moves, "
                                  f"{rate:.1f} pos/sec")
                        
                        # Commit SQLite periodically
                        if conn and games_processed % 50 == 0:
                            conn.commit()
                    
                    except Exception as e:
                        print(f"Error processing game {game_id}: {e}")
                        if conn:
                            conn.rollback()
                        import traceback
                        traceback.print_exc()
                        continue
            
            # Final commit and close
            if conn:
                conn.commit()
            
            # Close Parquet writers
            for writer in self.parquet_writers.values():
                writer.close()
        
        finally:
            if conn:
                conn.close()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        stats = self.evaluator.get_stats()
        
        results = {
            'games_processed': games_processed,
            'games_skipped': games_skipped,
            'positions_evaluated': positions_evaluated,
            'possible_moves_written': possible_moves_written,
            'elapsed_seconds': elapsed,
            'positions_per_second': positions_evaluated / elapsed if elapsed > 0 else 0,
            'engine': self.engine_info,
            'output_locations': {
                'sqlite': self.db_path if self.write_sqlite else None,
                'parquet': str(self.output_dir / 'lc0' / self.engine_info['version']) if self.write_parquet else None,
            },
            'engine_stats': stats,
        }
        
        print(f"\n{'='*70}")
        print(f"COMPLETED: {pgn_path.name}")
        print(f"{'='*70}")
        print(f"  Games processed:     {games_processed:,}")
        print(f"  Games skipped:       {games_skipped:,}")
        print(f"  Positions:           {positions_evaluated:,}")
        print(f"  Possible moves:      {possible_moves_written:,}")
        print(f"  Time:                {elapsed/60:.1f} minutes")
        print(f"  Rate:                {results['positions_per_second']:.1f} pos/sec")
        if self.write_parquet:
            print(f"  Parquet:             {results['output_locations']['parquet']}")
        if self.write_sqlite:
            print(f"  SQLite:              {results['output_locations']['sqlite']}")
        print(f"{'='*70}\n")
        
        return results
    
    def _process_game(self, conn: sqlite3.Connection, game: chess.pgn.Game, game_id: str) -> Dict:
        """
        Process single game and write to BOTH SQLite and Parquet
        
        CRITICAL: All three tables get engine annotations:
        - games.evaluated_by, games.evaluator_version
        - actual_moves.evaluated_by, actual_moves.evaluator_version  
        - possible_move_evals.evaluated_by, possible_move_evals.evaluator_version
        """
        headers = game.headers
        
        # Prepare game record WITH ENGINE TRACKING
        game_data = {
            'game_id': game_id,
            'white': headers.get('White', 'Unknown')[:100],
            'black': headers.get('Black', 'Unknown')[:100],
            'white_elo': self._safe_int(headers.get('WhiteElo')),
            'black_elo': self._safe_int(headers.get('BlackElo')),
            'result': headers.get('Result', '*'),
            'event': headers.get('Event', '')[:200],
            'site': headers.get('Site', '')[:200],
            'game_date': headers.get('Date', ''),
            'round': headers.get('Round', '')[:50],
            'eco': headers.get('ECO', ''),
            'time_control': headers.get('TimeControl', ''),
            'game_hash': game_id,
            # ENGINE TRACKING FIELDS
            'evaluated_by': 'lc0',
            'evaluator_version': self.engine_info['version'],
            'evaluated_at': self.engine_info['evaluated_at'],
            'pgn_text': str(game)[:10000],
        }
        
        # Write to SQLite
        if conn:
            conn.execute("""
                INSERT OR REPLACE INTO games (
                    game_id, white, black, white_elo, black_elo, result,
                    event, site, game_date, round, eco, time_control,
                    game_hash, evaluated_by, evaluator_version, evaluated_at, pgn_text
                ) VALUES (
                    :game_id, :white, :black, :white_elo, :black_elo, :result,
                    :event, :site, :game_date, :round, :eco, :time_control,
                    :game_hash, :evaluated_by, :evaluator_version, :evaluated_at, :pgn_text
                )
            """, game_data)
        
        # Write to Parquet
        if self.write_parquet:
            self.parquet_writers['games'].write(game_data)
        
        # Process moves
        board = game.board()
        move_number = 0
        positions = 0
        possible_moves_count = 0
        
        node = game
        prev_boards = []
        
        while node.variations:
            node = node.variation(0)
            move = node.move
            move_number += 1
            
            # Evaluate position BEFORE move
            eval_before = self.evaluator.evaluate_position(board)
            
            # Get legal moves and evaluate them for possible_move_evals table
            legal_moves = list(board.legal_moves)
            
            # Evaluate each legal move and write to possible_move_evals
            # THIS IS WHERE ENGINE ANNOTATION IS CRITICAL
            for legal_move in legal_moves[:30]:  # Limit to top 30 for speed
                board_copy = board.copy()
                board_copy.push(legal_move)
                eval_result = self.evaluator.evaluate_position(board_copy)
                
                possible_move_data = {
                    'game_id': game_id,
                    'move_no': move_number,
                    'move_uci': legal_move.uci(),
                    'centipawn': eval_result['ev'],
                    'mate_score': None,
                    'win_prob_white': eval_result['wdl'][0] / 10.0,
                    'win_prob_draw': eval_result['wdl'][1] / 10.0,
                    'win_prob_black': eval_result['wdl'][2] / 10.0,
                    'nodes': 1,
                    'depth': 1,
                    'pv': '',
                    # ENGINE TRACKING - CRITICAL FOR FILTERING
                    'evaluated_by': 'lc0',
                    'evaluator_version': self.engine_info['version'],
                }
                
                # Write to SQLite
                if conn:
                    conn.execute("""
                        INSERT INTO possible_move_evals (
                            game_id, move_no, move_uci, centipawn, mate_score,
                            win_prob_white, win_prob_draw, win_prob_black,
                            nodes, depth, pv, evaluated_by, evaluator_version
                        ) VALUES (
                            :game_id, :move_no, :move_uci, :centipawn, :mate_score,
                            :win_prob_white, :win_prob_draw, :win_prob_black,
                            :nodes, :depth, :pv, :evaluated_by, :evaluator_version
                        )
                    """, possible_move_data)
                
                # Write to Parquet
                if self.write_parquet:
                    self.parquet_writers['possible_moves'].write(possible_move_data)
                
                possible_moves_count += 1
            
            # Generate planes
            planes = board_to_planes(board, prev_boards)
            planes_blob = pack_planes(planes)
            
            # Make move and evaluate AFTER position
            board.push(move)
            eval_after = self.evaluator.evaluate_position(board)
            
            planes_after = board_to_planes(board, [board])
            planes_after_blob = pack_planes(planes_after)
            
            # Prepare actual move record WITH ENGINE TRACKING
            move_data = {
                'game_id': game_id,
                'move_no': move_number,
                'move_uci': move.uci(),
                'fen_before': board.fen(),
                'fen_after': board.fen(),
                'eval_before': eval_before['ev'],
                'eval_after': eval_after['ev'],
                'win_prob_white_before': eval_before['wdl'][0] / 10.0,
                'win_prob_draw_before': eval_before['wdl'][1] / 10.0,
                'win_prob_black_before': eval_before['wdl'][2] / 10.0,
                'win_prob_white_after': eval_after['wdl'][0] / 10.0,
                'win_prob_draw_after': eval_after['wdl'][1] / 10.0,
                'win_prob_black_after': eval_after['wdl'][2] / 10.0,
                'planes_before': planes_blob,
                'planes_after': planes_after_blob,
                'time_spent_ms': self._get_time_spent(node),
                'time_remaining_ms': self._get_time_remaining(node),
                # ENGINE TRACKING - CRITICAL FOR FILTERING
                'evaluated_by': 'lc0',
                'evaluator_version': self.engine_info['version'],
            }
            
            # Write to SQLite
            if conn:
                conn.execute("""
                    INSERT INTO actual_moves (
                        game_id, move_no, move_uci, fen_before, fen_after,
                        eval_before, eval_after,
                        win_prob_white_before, win_prob_draw_before, win_prob_black_before,
                        win_prob_white_after, win_prob_draw_after, win_prob_black_after,
                        planes_before, planes_after,
                        time_spent_ms, time_remaining_ms,
                        evaluated_by, evaluator_version
                    ) VALUES (
                        :game_id, :move_no, :move_uci, :fen_before, :fen_after,
                        :eval_before, :eval_after,
                        :win_prob_white_before, :win_prob_draw_before, :win_prob_black_before,
                        :win_prob_white_after, :win_prob_draw_after, :win_prob_black_after,
                        :planes_before, :planes_after,
                        :time_spent_ms, :time_remaining_ms,
                        :evaluated_by, :evaluator_version
                    )
                """, move_data)
            
            # Write to Parquet (excluding blobs for size)
            if self.write_parquet:
                # Create copy without blobs for Parquet (they're large)
                parquet_move_data = {k: v for k, v in move_data.items() 
                                   if k not in ['planes_before', 'planes_after']}
                self.parquet_writers['moves'].write(parquet_move_data)
            
            positions += 1
            prev_boards = [board.copy()] + prev_boards[:1]
        
        return {
            'positions': positions,
            'possible_moves': possible_moves_count,
        }
    
    def _safe_int(self, val) -> Optional[int]:
        try:
            return int(val) if val and val != '?' else None
        except:
            return None
    
    def _get_time_spent(self, node) -> Optional[int]:
        try:
            if hasattr(node, 'comment') and node.comment:
                import re
                match = re.search(r'\[%emt (\d+):(\d+):(\d+)\]', node.comment)
                if match:
                    h, m, s = map(int, match.groups())
                    return (h * 3600 + m * 60 + s) * 1000
            return None
        except:
            return None
    
    def _get_time_remaining(self, node) -> Optional[int]:
        try:
            if hasattr(node, 'comment') and node.comment:
                import re
                match = re.search(r'\[%clk (\d+):(\d+):(\d+)\]', node.comment)
                if match:
                    h, m, s = map(int, match.groups())
                    return (h * 3600 + m * 60 + s) * 1000
            return None
        except:
            return None
    
    def close(self):
        """Clean up resources"""
        for writer in self.parquet_writers.values():
            writer.close()
        self.evaluator.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LC0 processor with SQLite + Parquet output'
    )
    parser.add_argument('pgn_file', help='PGN file to process')
    parser.add_argument('--db', default='chess.db', help='SQLite database')
    parser.add_argument('--output-dir', default='output', help='Parquet output directory')
    parser.add_argument('--weights', required=True, help='LC0 weights file')
    parser.add_argument('--backend', default='cuda-fp16', help='LC0 backend')
    parser.add_argument('--no-parquet', action='store_true', help='Disable Parquet output')
    parser.add_argument('--no-sqlite', action='store_true', help='Disable SQLite output')
    parser.add_argument('--max-games', type=int, help='Max games to process')
    
    args = parser.parse_args()
    
    processor = LC0GameProcessorWithParquet(
        db_path=args.db,
        weights_path=args.weights,
        output_dir=args.output_dir,
        backend=args.backend,
        write_parquet=not args.no_parquet,
        write_sqlite=not args.no_sqlite,
        verbose=True
    )
    
    try:
        results = processor.process_pgn_file(args.pgn_file, max_games=args.max_games)
        
        print("\n" + "="*70)
        print("TRAINING DATA READY")
        print("="*70)
        print("\nTo train on LC0 data only:")
        print()
        print("From SQLite:")
        print("  SELECT * FROM actual_moves WHERE evaluated_by = 'lc0'")
        print()
        print("From Parquet:")
        print(f"  import pandas as pd")
        print(f"  df = pd.read_parquet('{results['output_locations']['parquet']}/moves/')")
        print(f"  df_lc0 = df[df['evaluated_by'] == 'lc0']")
        print()
        print("Filter by version:")
        print(f"  WHERE evaluator_version = '{results['engine']['version']}'")
        print("="*70)
        
    finally:
        processor.close()
