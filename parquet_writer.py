#!/usr/bin/env python3
"""
parquet_writer.py

Compatible with lc0_processor_with_parquet.py
Uses typed dataclass records with add_game/add_move/add_possible_move.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


@dataclass
class GameRecord:
    """Game metadata record — matches parquet games schema"""
    game_id: str = None
    game_order: int = None
    event: Optional[str] = None
    site: Optional[str] = None
    date_played: Optional[str] = None
    round: Optional[str] = None
    white: Optional[str] = None           # SHA-256 hashed
    black: Optional[str] = None           # SHA-256 hashed
    result: Optional[str] = None
    white_elo: Optional[int] = None
    white_rating_diff: Optional[int] = None
    black_elo: Optional[int] = None
    black_rating_diff: Optional[int] = None
    white_title: Optional[str] = None
    black_title: Optional[str] = None
    winner: Optional[str] = None
    winner_elo: Optional[int] = None
    loser: Optional[str] = None
    loser_elo: Optional[int] = None
    winner_loser_elo_diff: Optional[int] = None
    eco: Optional[str] = None
    termination: Optional[str] = None
    time_control: Optional[str] = None
    utc_date: Optional[str] = None
    utc_time: Optional[str] = None
    variant: Optional[str] = None
    ply_count: Optional[int] = None
    game_hash: Optional[str] = None
    evaluated_by: str = 'lc0'
    evaluator_version: str = 'unknown'
    evaluated_at: Optional[str] = None
    pgn_text: Optional[str] = None


@dataclass
class MoveRecord:
    """Actual move record — matches parquet moves schema"""
    game_id: str = None
    move_no: int = None
    move_no_pair: int = None
    player: Optional[str] = None          # SHA-256 hashed
    notation: Optional[str] = None        # SAN
    move: Optional[str] = None            # UCI
    from_square: Optional[str] = None
    to_square: Optional[str] = None
    piece: Optional[str] = None
    promotion: Optional[str] = None
    color: Optional[str] = None
    fen_before: Optional[str] = None
    fen_after: Optional[str] = None
    time_remaining: Optional[float] = None
    time_spent: Optional[float] = None
    game_to_position: Optional[str] = None
    white_win_perc_before: Optional[float] = None
    black_win_perc_before: Optional[float] = None
    draw_perc_before: Optional[float] = None
    white_win_perc_after: Optional[float] = None
    black_win_perc_after: Optional[float] = None
    draw_perc_after: Optional[float] = None
    static_eval_before: Optional[float] = None
    static_eval_after: Optional[float] = None
    eval_before: Optional[float] = None
    mate_count_before: Optional[float] = None
    eval_after: Optional[float] = None
    mate_count_after: Optional[float] = None
    evaluated_by: str = 'lc0'
    evaluator_version: str = 'unknown'


@dataclass
class PossibleMoveRecord:
    """Possible move record — matches parquet possible_moves schema"""
    game_id: str = None
    move_no: int = None
    move_no_pair: int = None
    notation: Optional[str] = None        # SAN
    move: Optional[str] = None            # UCI
    from_square: Optional[str] = None
    to_square: Optional[str] = None
    piece: Optional[str] = None
    promotion: Optional[str] = None
    color: Optional[str] = None
    fen_before: Optional[str] = None
    fen_after: Optional[str] = None
    eval: Optional[float] = None
    mate_count: Optional[float] = None
    white_win_perc: Optional[float] = None
    black_win_perc: Optional[float] = None
    draw_perc: Optional[float] = None
    evaluated_by: str = 'lc0'
    evaluator_version: str = 'unknown'


class ParquetWriter:
    """Writes chess evaluation data to Parquet files"""

    def __init__(self,
                 output_dir: str,
                 schema_type: str = 'games',
                 batch_size: int = 10000,
                 max_rows_per_file: int = 100000,
                 **kwargs):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.schema_type = schema_type
        self.batch_size = batch_size
        self.max_rows_per_file = max_rows_per_file

        self.batch = []
        self.file_counter = 0
        self.rows_in_current_file = 0

        if schema_type == 'games':
            self.filename_prefix = 'games'
        elif schema_type == 'moves':
            self.filename_prefix = 'moves'
        elif schema_type == 'possible_moves':
            self.filename_prefix = 'possible_moves'
        else:
            raise ValueError(f"Unknown schema_type: {schema_type}")

    def add_game(self, game: GameRecord):
        self.batch.append(game)
        if len(self.batch) >= self.batch_size:
            self._flush()

    def add_move(self, move: MoveRecord):
        self.batch.append(move)
        if len(self.batch) >= self.batch_size:
            self._flush()

    def add_possible_move(self, pm: PossibleMoveRecord):
        self.batch.append(pm)
        if len(self.batch) >= self.batch_size:
            self._flush()

    def _flush(self):
        if not self.batch:
            return

        df = pd.DataFrame([r.__dict__ for r in self.batch])
        output_file = self.output_dir / f"{self.filename_prefix}_{self.file_counter:06d}.parquet"

        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file, compression='ZSTD')

        print(f"Wrote {len(self.batch)} {self.schema_type} to {output_file.name}")
        self.rows_in_current_file += len(self.batch)
        self.batch = []

        if self.rows_in_current_file >= self.max_rows_per_file:
            self.file_counter += 1
            self.rows_in_current_file = 0

    def flush_all(self):
        self._flush()

    def close(self):
        self.flush_all()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
