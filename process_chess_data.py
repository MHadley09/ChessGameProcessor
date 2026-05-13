#!/usr/bin/env python3
"""
process_chess_data.py

Cross-platform chess data processor launcher.
Works on Windows, macOS, and Linux.

Handles all the setup, validation, and orchestration for you.

Usage:
    python process_chess_data.py /path/to/pgn/files/
    python process_chess_data.py C:\\data\\pgn\\ --max-games 1000
    
Environment variables:
    DB_PATH, OUTPUT_DIR, WEIGHTS_PATH, BACKEND, MAX_GAMES
"""

import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path
from datetime import datetime
import shutil


class ChessProcessorOrchestrator:
    """Orchestrates chess data processing with full cross-platform support"""
    
    def __init__(self):
        self.platform = platform.system()
        self.is_windows = self.platform == 'Windows'
        self.is_mac = self.platform == 'Darwin'
        self.is_linux = self.platform == 'Linux'
        
        # Colors (work on all platforms with colorama, but use plain on Windows by default)
        self.colors = {
            'GREEN': '\033[0;32m' if not self.is_windows else '',
            'RED': '\033[0;31m' if not self.is_windows else '',
            'YELLOW': '\033[1;33m' if not self.is_windows else '',
            'BLUE': '\033[0;34m' if not self.is_windows else '',
            'NC': '\033[0m' if not self.is_windows else '',
        }
    
    def log(self, msg):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{self.colors['GREEN']}[{timestamp}]{self.colors['NC']} {msg}")
    
    def warn(self, msg):
        print(f"{self.colors['YELLOW']}[WARNING]{self.colors['NC']} {msg}")
    
    def error(self, msg):
        print(f"{self.colors['RED']}[ERROR]{self.colors['NC']} {msg}", file=sys.stderr)
        sys.exit(1)
    
    def info(self, msg):
        print(f"{self.colors['BLUE']}[INFO]{self.colors['NC']} {msg}")
    
    def check_requirements(self, weights_path, python_script):
        """Check all requirements are installed"""
        self.log("Checking requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.error(f"Python 3.8+ required, found {sys.version}")
        
        # Check packages
        try:
            import chess
        except ImportError:
            self.error("python-chess not installed. Run: pip install chess")
        
        try:
            import pyarrow
        except ImportError:
            self.warn("pyarrow not installed. Parquet output disabled. Run: pip install pyarrow")
        
        # Check weights file
        if not Path(weights_path).exists():
            self.error(f"Weights file not found: {weights_path}\n"
                      f"Download from: https://lczero.org/play/networks\n"
                      f"Recommended: 703810.pb.gz (15MB, fast)")
        
        # Check processor script
        if not Path(python_script).exists():
            self.error(f"Processor script not found: {python_script}\n"
                      f"Make sure you've extracted lc0_processor_v2_with_parquet.tar.gz")
        
        # Check GPU (optional)
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'NVIDIA' in line or 'MiB' in line:
                        self.log(f"GPU detected: {line.strip()}")
                        break
        except FileNotFoundError:
            self.warn("nvidia-smi not found. LC0 will run on CPU (much slower)")
            self.warn("Install NVIDIA drivers and CUDA for GPU acceleration")
        
        self.log("Requirements check passed ✓")
    
    def init_database(self, db_path):
        """Initialize database with engine tracking"""
        self.log("Initializing database...")
        
        # Backup if exists
        if Path(db_path).exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(db_path).with_suffix(f'.backup.{timestamp}')
            self.info(f"Backing up existing database to {backup_path}")
            shutil.copy2(db_path, backup_path)
        
        # Check if engine tracking columns exist
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(games)")
        cols = [row[1] for row in cursor.fetchall()]
        
        if 'evaluated_by' not in cols:
            self.log("Adding engine tracking columns...")
            if Path('004_add_engine_tracking.sql').exists():
                with open('004_add_engine_tracking.sql', 'r') as f:
                    conn.executescript(f.read())
                self.log("Engine tracking columns added ✓")
            else:
                self.error("Migration file not found: 004_add_engine_tracking.sql")
        else:
            self.log("Engine tracking columns already exist ✓")
        
        conn.close()
    
    def process_directory(self, pgn_dir, db_path, output_dir, weights_path, 
                         backend, max_games, log_dir):
        """Process all PGN files in directory"""
        pgn_dir = Path(pgn_dir)
        
        # Find PGN files
        pgn_files = list(pgn_dir.glob('**/*.pgn')) + list(pgn_dir.glob('**/*.pgn.gz'))
        
        if not pgn_files:
            self.error(f"No PGN files found in {pgn_dir}")
        
        self.log(f"Found {len(pgn_files)} PGN files to process")
        
        # Create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Process each file
        completed = 0
        failed = 0
        
        for pgn_file in pgn_files:
            log_file = Path(log_dir) / f"{pgn_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            self.log(f"Processing [{completed+1}/{len(pgn_files)}]: {pgn_file.name}")
            
            # Build command
            cmd = [
                sys.executable,
                str(Path('lc0_processor_with_parquet.py').absolute()),
                str(pgn_file),
                '--db', str(db_path),
                '--output-dir', str(output_dir),
                '--weights', str(weights_path),
                '--backend', backend,
            ]
            
            if max_games:
                cmd.extend(['--max-games', str(max_games)])
            
            # Run with logging
            try:
                with open(log_file, 'w') as log_f:
                    result = subprocess.run(
                        cmd,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=True
                    )
                completed += 1
                self.log(f"Completed: {pgn_file.name} ✓")
                
            except subprocess.CalledProcessError as e:
                self.warn(f"Failed: {pgn_file.name} - Check log: {log_file}")
                failed += 1
                # Continue with next file
                continue
        
        # Summary
        print()
        self.log("Processing complete!")
        self.log(f"  Completed: {completed}")
        self.log(f"  Failed: {failed}")
        self.log(f"  Total: {len(pgn_files)}")
        
        if failed > 0:
            self.warn(f"{failed} files failed. Check logs in {log_dir}")
            return 1
        
        return 0
    
    def show_summary(self, db_path, output_dir):
        """Show database and output summary"""
        self.log("Database Summary:")
        print()
        
        if Path(db_path).exists():
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("""
                SELECT 
                    'Total games' as metric,
                    CAST(COUNT(*) AS VARCHAR)
                FROM games
                UNION ALL
                SELECT 
                    'Games by engine',
                    evaluated_by || ': ' || COUNT(*)
                FROM games
                GROUP BY evaluated_by
                UNION ALL
                SELECT
                    'Total positions',
                    CAST(COUNT(*) AS VARCHAR)
                FROM actual_moves
                UNION ALL
                SELECT
                    'Positions by engine',
                    evaluated_by || ': ' || COUNT(*)
                FROM actual_moves
                GROUP BY evaluated_by
            """)
            
            for row in cursor.fetchall():
                print(f"  {row[0]}: {row[1]}")
            conn.close()
        else:
            self.warn(f"Database not found: {db_path}")
        
        print()
        self.log("Output locations:")
        print(f"  SQLite: {db_path}")
        print(f"  Parquet: {output_dir}")
        
        if Path(output_dir).exists():
            parquet_files = list(Path(output_dir).rglob('*.parquet'))
            print(f"  Parquet files: {len(parquet_files)}")
            
            # Show total size
            total_size = sum(f.stat().st_size for f in parquet_files)
            print(f"  Total size: {total_size / (1024**3):.2f} GB")
    
    def run(self, args):
        """Main execution"""
        print("="*70)
        print(f"Chess Data Processor - {self.platform}")
        print("="*70)
        print(f"PGN directory: {args.pgn_dir}")
        print(f"Database: {args.db}")
        print(f"Output: {args.output_dir}")
        print(f"Weights: {args.weights}")
        print(f"Backend: {args.backend}")
        print(f"Max games: {args.max_games or 'unlimited'}")
        print()
        
        self.check_requirements(args.weights, 'lc0_processor_with_parquet.py')
        print()
        self.init_database(args.db)
        print()
        
        exit_code = self.process_directory(
            args.pgn_dir,
            args.db,
            args.output_dir,
            args.weights,
            args.backend,
            args.max_games,
            args.log_dir
        )
        
        print()
        self.show_summary(args.db, args.output_dir)
        
        print()
        self.log("Pipeline completed successfully! ✓")
        print()
        self.info("Next steps:")
        self.info(f"  1. Verify: sqlite3 {args.db} \"SELECT COUNT(*) FROM games;\"")
        self.info(f"  2. Train: Use data from {args.output_dir} or {args.db}")
        self.info(f"  3. Filter: WHERE evaluated_by = 'lc0'")
        
        return exit_code


def main():
    parser = argparse.ArgumentParser(
        description='Cross-platform chess data processor (Windows/Mac/Linux)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python process_chess_data.py /path/to/pgn/
  python process_chess_data.py C:\\data\\pgn\\
  
  # With options
  python process_chess_data.py /data/pgn/ --max-games 1000 --weights weights/703810.pb.gz
  
  # Using environment variables
  DB_PATH=/data/chess.db WEIGHTS_PATH=weights/703810.pb.gz python process_chess_data.py /data/pgn/
        """
    )
    
    parser.add_argument('pgn_dir', help='Directory containing PGN files')
    parser.add_argument('--db', default=os.environ.get('DB_PATH', './chess.db'),
                       help='SQLite database path')
    parser.add_argument('--output-dir', default=os.environ.get('OUTPUT_DIR', './output'),
                       help='Output directory for Parquet files')
    parser.add_argument('--weights', default=os.environ.get('WEIGHTS_PATH', './weights/703810.pb.gz'),
                       help='LC0 weights file path')
    parser.add_argument('--backend', default=os.environ.get('BACKEND', 'cuda-fp16'),
                       choices=['cuda', 'cuda-fp16', 'cudnn', 'cudnn-fp16', 'opencl', 'blas'],
                       help='LC0 backend')
    parser.add_argument('--max-games', type=int, 
                       default=int(os.environ.get('MAX_GAMES', '0')) or None,
                       help='Maximum games per file (default: all)')
    parser.add_argument('--log-dir', default=os.environ.get('LOG_DIR', './logs'),
                       help='Directory for log files')
    
    args = parser.parse_args()
    
    # Validate
    if not Path(args.pgn_dir).exists():
        print(f"Error: Directory not found: {args.pgn_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Run
    orchestrator = ChessProcessorOrchestrator()
    exit_code = orchestrator.run(args)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
