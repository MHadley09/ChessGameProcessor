#!/usr/bin/env python3
"""
deduplication_helpers.py

Helper functions for game and position deduplication.
Import these into your existing processing code WITHOUT changing evaluation logic.

Usage:
    from deduplication_helpers import GameDeduplicator, PositionCache
    
    dedup = GameDeduplicator('chess.db')
    
    for game in games:
        game_hash = dedup.get_game_hash(game)
        
        if dedup.game_exists(game_hash):
            print(f"Skipping duplicate game {game_hash}")
            continue
        
        # Process game normally with your existing evaluation code
        process_game(game)
        
        # Mark as processed
        dedup.mark_game_processed(game_hash, game_id)
"""

import hashlib
import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any
import chess.pgn


class GameDeduplicator:
    """Handle game-level deduplication"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Ensure required tables and columns exist"""
        cursor = self.conn.cursor()
        
        # Add game_hash column if not exists
        cursor.execute("PRAGMA table_info(games)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'game_hash' not in columns:
            cursor.execute("ALTER TABLE games ADD COLUMN game_hash TEXT")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_hash ON games(game_hash)")
            self.conn.commit()
    
    def get_game_hash(self, game: chess.pgn.Game) -> str:
        """
        Generate hash for game deduplication.
        
        Uses: White, Black, Date, Event, and move list
        This is deterministic - same game always gets same hash.
        """
        # Extract headers
        headers = game.headers
        white = headers.get('White', 'Unknown')
        black = headers.get('Black', 'Unknown')
        date = headers.get('Date', '')
        event = headers.get('Event', '')
        site = headers.get('Site', '')
        round_num = headers.get('Round', '')
        
        # Get moves as string
        moves = []
        node = game
        while node.variations:
            node = node.variations[0]
            moves.append(node.move.uci())
        
        moves_str = ' '.join(moves)
        
        # Create hash input - include enough to be unique but stable
        # Don't include headers that might vary (like annotations)
        hash_input = f"{white}|{black}|{date}|{event}|{site}|{round_num}|{moves_str}"
        
        # Return first 16 characters of SHA256
        # 16 hex chars = 64 bits = ~1.8e19 possibilities
        # For 10M games, collision probability is ~0.0000003%
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()[:16]
    
    def game_exists(self, game_hash: str) -> Optional[int]:
        """
        Check if game already exists.
        Returns game_id if exists, None otherwise.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT game_id FROM games WHERE game_hash = ? LIMIT 1",
            [game_hash]
        )
        result = cursor.fetchone()
        return result[0] if result else None
    
    def mark_game_processed(self, game_hash: str, game_id: int):
        """Update game record with its hash"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE games SET game_hash = ? WHERE game_id = ?",
            [game_hash, game_id]
        )
        self.conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM games")
        total_games = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM games WHERE game_hash IS NOT NULL")
        hashed_games = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT game_hash FROM games 
                WHERE game_hash IS NOT NULL 
                GROUP BY game_hash 
                HAVING COUNT(*) > 1
            )
        """)
        duplicate_groups = cursor.fetchone()[0]
        
        return {
            'total_games': total_games,
            'hashed_games': hashed_games,
            'unhashed_games': total_games - hashed_games,
            'duplicate_groups': duplicate_groups,
            'hash_coverage': hashed_games / total_games if total_games > 0 else 0
        }
    
    def close(self):
        self.conn.close()


class PositionCache:
    """
    Cache for position evaluations to avoid re-evaluating the same positions.
    
    This is SEPARATE from game deduplication - it caches the Stockfish
    evaluations themselves, not the games.
    
    Usage:
        cache = PositionCache('positions_cache.db')
        
        # Before evaluating
        cached = cache.get(fen)
        if cached:
            return cached
        
        # Evaluate with Stockfish (your existing code)
        evaluation = stockfish.get_evaluation()
        
        # Save to cache
        cache.set(fen, evaluation)
    """
    
    def __init__(self, db_path: str = "positions_cache.db", ttl_days: int = 30):
        self.db_path = db_path
        self.ttl_days = ttl_days
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize cache database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_evaluations (
                position_hash TEXT PRIMARY KEY,
                fen TEXT NOT NULL,
                evaluation_json TEXT NOT NULL,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_fen ON position_evaluations(fen)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_accessed ON position_evaluations(last_accessed)"
        )
        self.conn.commit()
    
    def _get_position_hash(self, fen: str) -> str:
        """Get hash for position (ignoring move counters)"""
        # Use position only, not full FEN (which includes move numbers)
        parts = fen.split(' ')
        position_key = ' '.join(parts[:4])  # board, turn, castling, en passant
        return hashlib.sha256(position_key.encode()).hexdigest()[:16]
    
    def get(self, fen: str) -> Optional[Dict]:
        """Get cached evaluation for position"""
        pos_hash = self._get_position_hash(fen)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT evaluation_json FROM position_evaluations
            WHERE position_hash = ?
            AND evaluated_at > datetime('now', '-{} days')
        """.format(self.ttl_days), [pos_hash])
        
        result = cursor.fetchone()
        if result:
            # Update access stats
            cursor.execute("""
                UPDATE position_evaluations
                SET access_count = access_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE position_hash = ?
            """, [pos_hash])
            self.conn.commit()
            
            return json.loads(result[0])
        
        return None
    
    def set(self, fen: str, evaluation: Dict):
        """Cache evaluation for position"""
        pos_hash = self._get_position_hash(fen)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO position_evaluations
            (position_hash, fen, evaluation_json, evaluated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, [pos_hash, fen, json.dumps(evaluation)])
        self.conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM position_evaluations")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(access_count) FROM position_evaluations")
        total_accesses = cursor.fetchone()[0] or 0
        
        cursor.execute("""
            SELECT COUNT(*) FROM position_evaluations
            WHERE evaluated_at > datetime('now', '-7 days')
        """)
        recent = cursor.fetchone()[0]
        
        return {
            'total_positions': total,
            'total_accesses': total_accesses,
            'recent_positions': recent,
            'avg_accesses_per_position': total_accesses / total if total > 0 else 0
        }
    
    def cleanup_old_entries(self, days: int = 90):
        """Remove entries older than specified days"""
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM position_evaluations
            WHERE evaluated_at < datetime('now', '-{} days')
            AND access_count < 5
        """.format(days))
        deleted = cursor.rowcount
        self.conn.commit()
        return deleted
    
    def close(self):
        self.conn.close()


class ProcessingTracker:
    """
    Track which files have been processed to enable incremental processing.
    
    Usage:
        tracker = ProcessingTracker('chess.db')
        
        files_to_process = tracker.filter_unprocessed([
            'games1.pgn',
            'games2.pgn',
            'games3.pgn'
        ])
        
        for file_path in files_to_process:
            # Process file
            stats = process_file(file_path)
            
            # Mark as complete
            tracker.mark_complete(file_path, stats)
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_log (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                games_found INTEGER DEFAULT 0,
                games_processed INTEGER DEFAULT 0,
                games_skipped INTEGER DEFAULT 0,
                processing_started TIMESTAMP,
                processing_completed TIMESTAMP,
                status TEXT DEFAULT 'pending',
                error_message TEXT
            )
        """)
        self.conn.commit()
    
    def _get_file_hash(self, file_path: str) -> str:
        """Compute hash of file contents"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def should_process(self, file_path: str) -> bool:
        """
        Check if file should be processed.
        Returns True if file is new or changed, False if already processed.
        """
        if not Path(file_path).exists():
            return False
        
        current_hash = self._get_file_hash(file_path)
        
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT file_hash FROM processing_log WHERE file_path = ? AND status = 'complete'",
            [file_path]
        )
        result = cursor.fetchone()
        
        # Process if not found or hash differs
        return not result or result[0] != current_hash
    
    def filter_unprocessed(self, file_paths: list) -> list:
        """Filter list to only files that need processing"""
        return [fp for fp in file_paths if self.should_process(fp)]
    
    def mark_started(self, file_path: str):
        """Mark file as processing started"""
        file_hash = self._get_file_hash(file_path)
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO processing_log
            (file_path, file_hash, processing_started, status)
            VALUES (?, ?, datetime('now'), 'processing')
        """, [file_path, file_hash])
        self.conn.commit()
    
    def mark_complete(self, file_path: str, stats: Dict[str, int]):
        """Mark file as successfully processed"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE processing_log
            SET games_found = ?,
                games_processed = ?,
                games_skipped = ?,
                processing_completed = datetime('now'),
                status = 'complete',
                error_message = NULL
            WHERE file_path = ?
        """, [
            stats.get('found', 0),
            stats.get('processed', 0),
            stats.get('skipped', 0),
            file_path
        ])
        self.conn.commit()
    
    def mark_failed(self, file_path: str, error: str):
        """Mark file as failed"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE processing_log
            SET processing_completed = datetime('now'),
                status = 'failed',
                error_message = ?
            WHERE file_path = ?
        """, [error, file_path])
        self.conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(games_processed) as total_games
            FROM processing_log
        """)
        
        row = cursor.fetchone()
        return {
            'total_files': row[0] or 0,
            'completed_files': row[1] or 0,
            'failed_files': row[2] or 0,
            'total_games_processed': row[3] or 0
        }
    
    def close(self):
        self.conn.close()


# Example integration with existing code
def example_usage():
    """
    Example of how to integrate deduplication into existing processing code
    WITHOUT changing evaluation logic.
    """
    # Initialize helpers
    dedup = GameDeduplicator('chess.db')
    tracker = ProcessingTracker('chess.db')
    # cache = PositionCache('positions_cache.db')  # Optional
    
    # Get list of PGN files to process
    import glob
    pgn_files = glob.glob('*.pgn')
    
    # Filter to unprocessed files
    files_to_process = tracker.filter_unprocessed(pgn_files)
    print(f"Found {len(files_to_process)} files to process")
    
    for file_path in files_to_process:
        print(f"\nProcessing {file_path}...")
        tracker.mark_started(file_path)
        
        stats = {'found': 0, 'processed': 0, 'skipped': 0}
        
        try:
            with open(file_path) as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    stats['found'] += 1
                    
                    # Check for duplicate BEFORE processing
                    game_hash = dedup.get_game_hash(game)
                    existing_id = dedup.game_exists(game_hash)
                    
                    if existing_id:
                        stats['skipped'] += 1
                        continue  # Skip duplicate
                    
                    # Process game with YOUR EXISTING CODE
                    # (No changes to evaluation logic needed)
                    game_id = process_game_with_your_existing_code(game)
                    
                    # Mark game with its hash
                    dedup.mark_game_processed(game_hash, game_id)
                    stats['processed'] += 1
                    
                    if stats['processed'] % 100 == 0:
                        print(f"  Processed {stats['processed']} new games...")
            
            tracker.mark_complete(file_path, stats)
            print(f"✓ Complete: {stats['processed']} new, {stats['skipped']} skipped")
            
        except Exception as e:
            tracker.mark_failed(file_path, str(e))
            print(f"✗ Failed: {e}")
    
    # Print final stats
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print("Deduplication:")
    for k, v in dedup.get_stats().items():
        print(f"  {k}: {v}")
    print("\nProcessing:")
    for k, v in tracker.get_stats().items():
        print(f"  {k}: {v}")
    
    dedup.close()
    tracker.close()


if __name__ == '__main__':
    example_usage()
