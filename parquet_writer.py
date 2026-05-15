"""
ParquetWriter — Writes game evaluation data to parquet files.

Imports schemas from parquet_schema.py (the canonical source of truth).
Dataclass fields match the schemas exactly (same names, same order).
Each flush produces a NEW file — no overwrites.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from parquet_schema import GAMES_SCHEMA, MOVES_SCHEMA, POSSIBLE_MOVES_SCHEMA


# ── Dataclasses matching parquet_schema.py 1:1 ──────────────────────────────

@dataclass
class GameRecord:
    """Fields match GAMES_SCHEMA exactly."""
    game_id: str = ""
    game_order: Optional[int] = None
    event: str = ""
    site: str = ""
    date_played: str = ""
    round: str = ""
    white: str = ""
    black: str = ""
    result: str = ""
    white_elo: int = 0
    white_rating_diff: Optional[int] = None
    black_elo: int = 0
    black_rating_diff: Optional[int] = None
    white_title: Optional[str] = None
    black_title: Optional[str] = None
    winner: Optional[str] = None
    winner_elo: Optional[int] = None
    loser: Optional[str] = None
    loser_elo: Optional[int] = None
    winner_loser_elo_diff: Optional[int] = None
    eco: str = ""
    termination: Optional[str] = None
    time_control: str = ""
    utc_date: Optional[str] = None
    utc_time: Optional[str] = None
    variant: Optional[str] = None
    ply_count: int = 0
    game_hash: str = ""
    evaluated_by: str = ""
    evaluator_version: str = ""
    evaluated_at: Optional[str] = None
    pgn_text: str = ""


@dataclass
class MoveRecord:
    """Fields match MOVES_SCHEMA exactly."""
    game_id: str = ""
    move_no: int = 0
    move_no_pair: int = 0
    player: str = ""
    notation: str = ""
    move: str = ""
    from_square: str = ""
    to_square: str = ""
    piece: str = ""
    promotion: Optional[str] = None
    color: str = ""
    fen_before: str = ""
    fen_after: str = ""
    time_remaining: Optional[float] = None
    time_spent: Optional[float] = None
    game_to_position: str = ""
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
    evaluated_by: str = ""
    evaluator_version: str = ""


@dataclass
class PossibleMoveRecord:
    """Fields match POSSIBLE_MOVES_SCHEMA exactly."""
    game_id: str = ""
    move_no: int = 0
    move_no_pair: int = 0
    notation: str = ""
    move: str = ""
    from_square: str = ""
    to_square: str = ""
    piece: str = ""
    promotion: Optional[str] = None
    color: str = ""
    fen_before: str = ""
    fen_after: str = ""
    eval: Optional[float] = None
    mate_count: Optional[float] = None
    white_win_perc: Optional[float] = None
    black_win_perc: Optional[float] = None
    draw_perc: Optional[float] = None
    nodes: int = 0
    depth: int = 0
    pv: Optional[str] = None
    evaluated_by: str = ""
    evaluator_version: str = ""


def _to_dict(record) -> dict:
    """Convert a dataclass or dict to dict."""
    if isinstance(record, dict):
        return record
    return asdict(record)


class ParquetWriter:
    """
    Buffered parquet writer that flushes batches to sequentially-numbered
    part files. Each flush produces a NEW file — no overwrites.

    Output structure:
        output_dir/
          games/
            part_000000.parquet
          moves/
            part_000000.parquet
          possible_moves/
            part_000000.parquet
    """

    def __init__(
        self,
        output_dir: str,
        games_batch_size: int = 10_000,
        moves_batch_size: int = 10_000,
        possible_moves_batch_size: int = 10_000,
        worker_id: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self._batch_sizes = {
            "games": games_batch_size,
            "moves": moves_batch_size,
            "possible_moves": possible_moves_batch_size,
        }
        self.worker_id = worker_id

        # Separate counters per table type
        self._file_counters = {"games": 0, "moves": 0, "possible_moves": 0}

        # Buffers
        self._game_buffer: List[dict] = []
        self._move_buffer: List[dict] = []
        self._possible_move_buffer: List[dict] = []

        # Totals
        self.games_written = 0
        self.moves_written = 0
        self.possible_moves_written = 0

        # Create dirs
        for sub in ("games", "moves", "possible_moves"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    # ── Public API ───────────────────────────────────────────────────────

    def write_game(self, record):
        self._game_buffer.append(_to_dict(record))
        if len(self._game_buffer) >= self._batch_sizes["games"]:
            self._flush_buffer("games", self._game_buffer, GAMES_SCHEMA)
            self._game_buffer = []

    def write_move(self, record):
        self._move_buffer.append(_to_dict(record))
        if len(self._move_buffer) >= self._batch_sizes["moves"]:
            self._flush_buffer("moves", self._move_buffer, MOVES_SCHEMA)
            self._move_buffer = []

    def write_possible_move(self, record):
        self._possible_move_buffer.append(_to_dict(record))
        if len(self._possible_move_buffer) >= self._batch_sizes["possible_moves"]:
            self._flush_buffer("possible_moves", self._possible_move_buffer, POSSIBLE_MOVES_SCHEMA)
            self._possible_move_buffer = []

    def flush(self):
        """Flush all remaining buffered data to disk."""
        if self._game_buffer:
            self._flush_buffer("games", self._game_buffer, GAMES_SCHEMA)
            self._game_buffer = []
        if self._move_buffer:
            self._flush_buffer("moves", self._move_buffer, MOVES_SCHEMA)
            self._move_buffer = []
        if self._possible_move_buffer:
            self._flush_buffer("possible_moves", self._possible_move_buffer, POSSIBLE_MOVES_SCHEMA)
            self._possible_move_buffer = []

    def close(self):
        """Flush and finalize."""
        self.flush()

    # ── Internal ─────────────────────────────────────────────────────────

    def _flush_buffer(self, table_name: str, buffer: List[dict], schema: pa.Schema):
        if not buffer:
            return

        # Replace None with appropriate zero for non-nullable columns
        for row in buffer:
            for key, val in row.items():
                if val is None:
                    try:
                        arrow_field = schema.field(key)
                    except KeyError:
                        continue
                    if arrow_field.type in (pa.int32(), pa.int64()):
                        row[key] = 0
                    elif arrow_field.type == pa.float64():
                        row[key] = 0.0

        table = pa.Table.from_pylist(buffer, schema=schema)

        # Each flush gets its own file number — never overwrite
        counter = self._file_counters[table_name]
        if self.worker_id is not None:
            filename = f"worker{self.worker_id:02d}_part_{counter:06d}.parquet"
        else:
            filename = f"part_{counter:06d}.parquet"
        output_path = self.output_dir / table_name / filename
        self._file_counters[table_name] = counter + 1

        pq.write_table(table, str(output_path), compression="snappy")

        count = len(buffer)
        if table_name == "games":
            self.games_written += count
        elif table_name == "moves":
            self.moves_written += count
        else:
            self.possible_moves_written += count

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
