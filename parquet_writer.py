#!/usr/bin/env python3
"""
parquet_writer.py

Compatible with lc0_processor_with_parquet.py
Accepts schema_type and max_rows_per_file parameters.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

@dataclass
class GameRecord:
    """Game metadata record"""
    game_id: int
    event: Optional[str] = None
    site: Optional[str] = None
    date_played: Optional[str] = None
    round: Optional[str] = None
    white: Optional[str] = None
    black: Optional[str] = None
    result: Optional[str] = None
    eco: Optional[str] = None
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    evaluated_by: str = 'lc0'
    evaluator_version: str = 'unknown'
    
@dataclass
class MoveRecord:
    """Individual move evaluation record"""
    game_id: int
    ply: int
    move_san: str
    move_uci: str
    fen_before: str
    fen_after: str
    eval_before: Optional[float] = None
    eval_after: Optional[float] = None
    wdl_before: Optional[List[int]] = None
    wdl_after: Optional[List[int]] = None
    best_move: Optional[str] = None
    evaluated_by: str = 'lc0'
    evaluator_version: str = 'unknown'
    
@dataclass
class PossibleMoveRecord:
    """Alternative move evaluations for a position"""
    game_id: int
    ply: int
    move_san: str
    move_uci: str
    centipawn: int
    mate: Optional[int] = None
    evaluated_by: str = 'lc0'
    evaluator_version: str = 'unknown'

class ParquetWriter:
    """Writes chess evaluation data to Parquet files"""
    
    def __init__(self, 
                 output_dir: str, 
                 schema_type: str = 'games', 
                 batch_size: int = 10000,
                 max_rows_per_file: int = 100000,
                 **kwargs):  # Accept any other params for compatibility
        """
        Args:
            output_dir: Directory to write Parquet files
            schema_type: Type of schema ('games', 'moves', or 'possible_moves')
            batch_size: Number of records to buffer before writing
            max_rows_per_file: Maximum rows per Parquet file before rotating
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.schema_type = schema_type
        self.batch_size = batch_size
        self.max_rows_per_file = max_rows_per_file
        
        self.batch = []
        self.file_counter = 0
        self.rows_in_current_file = 0
        
        # Define schemas for each type
        if schema_type == 'games':
            self.filename_prefix = 'games'
        elif schema_type == 'moves':
            self.filename_prefix = 'moves'
        elif schema_type == 'possible_moves':
            self.filename_prefix = 'possible_moves'
        else:
            raise ValueError(f"Unknown schema_type: {schema_type}")
    
    def add_game(self, game: GameRecord):
        """Add a game record to the batch"""
        if self.schema_type != 'games':
            raise TypeError(f"Writer initialized for '{self.schema_type}', cannot add GameRecord")
        self.batch.append(game)
        if len(self.batch) >= self.batch_size:
            self._flush()
    
    def add_move(self, move: MoveRecord):
        """Add a move record to the batch"""
        if self.schema_type != 'moves':
            raise TypeError(f"Writer initialized for '{self.schema_type}', cannot add MoveRecord")
        self.batch.append(move)
        if len(self.batch) >= self.batch_size:
            self._flush()
    
    def add_possible_move(self, pm: PossibleMoveRecord):
        """Add a possible move record to the batch"""
        if self.schema_type != 'possible_moves':
            raise TypeError(f"Writer initialized for '{self.schema_type}', cannot add PossibleMoveRecord")
        self.batch.append(pm)
        if len(self.batch) >= self.batch_size:
            self._flush()
    
    def add(self, record):
        """Generic add method that detects record type"""
        if isinstance(record, GameRecord):
            self.add_game(record)
        elif isinstance(record, MoveRecord):
            self.add_move(record)
        elif isinstance(record, PossibleMoveRecord):
            self.add_possible_move(record)
        else:
            raise TypeError(f"Unknown record type: {type(record)}")
    
    def _flush(self):
        """Write current batch to Parquet"""
        if not self.batch:
            return
            
        df = pd.DataFrame([r.__dict__ for r in self.batch])
        output_file = self.output_dir / f"{self.filename_prefix}_{self.file_counter:06d}.parquet"
        
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file, compression='ZSTD')
        
        print(f"Wrote {len(self.batch)} {self.schema_type} to {output_file.name}")
        self.rows_in_current_file += len(self.batch)
        self.batch = []
        
        # Rotate file if we've hit max_rows_per_file
        if self.rows_in_current_file >= self.max_rows_per_file:
            self.file_counter += 1
            self.rows_in_current_file = 0
    
    def flush_all(self):
        """Flush all remaining records"""
        self._flush()
    
    def close(self):
        """Close writer and flush all data"""
        self.flush_all()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()