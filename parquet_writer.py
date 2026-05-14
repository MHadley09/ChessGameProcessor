"""
ParquetWriter — Fixed version that doesn't overwrite data on each flush.

BUG FIX: The original _flush() called pq.write_table(table, output_file) which
OVERWRITES the file every time. With batch_size=10000 and max_rows_per_file=100000,
the first 9 flushes to _000000.parquet each destroyed the previous data. Only the
last 10K rows survived per file rotation.

FIX: Each flush increments file_counter so every batch lands in its own file.
This is simple, safe, and lets downstream readers use pyarrow.parquet.ParquetDataset
to read all part files transparently.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


# ── Schemas ──────────────────────────────────────────────────────────────────

@dataclass
class GameRecord:
    game_id: str = ""
    event: str = ""
    site: str = ""
    date: str = ""
    round: str = ""
    white: str = ""
    black: str = ""
    result: str = ""
    white_elo: int = 0
    black_elo: int = 0
    time_control: str = ""
    eco: str = ""
    opening: str = ""
    pgn_text: str = ""
    num_moves: int = 0
    evaluated_by: str = ""
    evaluator_version: str = ""


@dataclass
class MoveRecord:
    game_id: str = ""
    ply: int = 0
    fen: str = ""
    move_uci: str = ""
    move_san: str = ""
    score_cp: Optional[int] = None
    score_mate: Optional[int] = None
    top_move_uci: str = ""
    top_move_san: str = ""
    top_score_cp: Optional[int] = None
    top_score_mate: Optional[int] = None
    is_best_move: bool = False
    evaluated_by: str = ""


@dataclass
class PossibleMoveRecord:
    game_id: str = ""
    ply: int = 0
    fen: str = ""
    move_uci: str = ""
    move_san: str = ""
    score_cp: Optional[int] = None
    score_mate: Optional[int] = None
    rank: int = 0
    prior_probability: float = 0.0
    visits: int = 0
    evaluated_by: str = ""


# Arrow schemas for the three tables
GAME_SCHEMA = pa.schema([
    ("game_id", pa.string()),
    ("event", pa.string()),
    ("site", pa.string()),
    ("date", pa.string()),
    ("round", pa.string()),
    ("white", pa.string()),
    ("black", pa.string()),
    ("result", pa.string()),
    ("white_elo", pa.int32()),
    ("black_elo", pa.int32()),
    ("time_control", pa.string()),
    ("eco", pa.string()),
    ("opening", pa.string()),
    ("pgn_text", pa.string()),
    ("num_moves", pa.int32()),
    ("evaluated_by", pa.string()),
    ("evaluator_version", pa.string()),
])

MOVE_SCHEMA = pa.schema([
    ("game_id", pa.string()),
    ("ply", pa.int32()),
    ("fen", pa.string()),
    ("move_uci", pa.string()),
    ("move_san", pa.string()),
    ("score_cp", pa.int32()),
    ("score_mate", pa.int32()),
    ("top_move_uci", pa.string()),
    ("top_move_san", pa.string()),
    ("top_score_cp", pa.int32()),
    ("top_score_mate", pa.int32()),
    ("is_best_move", pa.bool_()),
    ("evaluated_by", pa.string()),
])

POSSIBLE_MOVE_SCHEMA = pa.schema([
    ("game_id", pa.string()),
    ("ply", pa.int32()),
    ("fen", pa.string()),
    ("move_uci", pa.string()),
    ("move_san", pa.string()),
    ("score_cp", pa.int32()),
    ("score_mate", pa.int32()),
    ("rank", pa.int32()),
    ("prior_probability", pa.float64()),
    ("visits", pa.int32()),
    ("evaluated_by", pa.string()),
])


class ParquetWriter:
    """
    Buffered parquet writer that flushes batches to sequentially-numbered
    part files.  Each flush produces a NEW file — no overwrites.

    Output structure:
        output_dir/
          games/
            part_000000.parquet
            part_000001.parquet
            ...
          moves/
            part_000000.parquet
            ...
          possible_moves/
            part_000000.parquet
            ...
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

    def write_game(self, record: GameRecord):
        self._game_buffer.append(asdict(record))
        if len(self._game_buffer) >= self._batch_sizes["games"]:
            self._flush_buffer("games", self._game_buffer, GAME_SCHEMA)
            self._game_buffer = []

    def write_move(self, record: MoveRecord):
        self._move_buffer.append(asdict(record))
        if len(self._move_buffer) >= self._batch_sizes["moves"]:
            self._flush_buffer("moves", self._move_buffer, MOVE_SCHEMA)
            self._move_buffer = []

    def write_possible_move(self, record: PossibleMoveRecord):
        self._possible_move_buffer.append(asdict(record))
        if len(self._possible_move_buffer) >= self._batch_sizes["possible_moves"]:
            self._flush_buffer("possible_moves", self._possible_move_buffer, POSSIBLE_MOVE_SCHEMA)
            self._possible_move_buffer = []

    def flush(self):
        """Flush all remaining buffered data to disk."""
        if self._game_buffer:
            self._flush_buffer("games", self._game_buffer, GAME_SCHEMA)
            self._game_buffer = []
        if self._move_buffer:
            self._flush_buffer("moves", self._move_buffer, MOVE_SCHEMA)
            self._move_buffer = []
        if self._possible_move_buffer:
            self._flush_buffer("possible_moves", self._possible_move_buffer, POSSIBLE_MOVE_SCHEMA)
            self._possible_move_buffer = []

    def close(self):
        """Flush and finalize."""
        self.flush()

    # ── Internal ─────────────────────────────────────────────────────────

    def _flush_buffer(self, table_name: str, buffer: List[dict], schema: pa.Schema):
        if not buffer:
            return

        # Replace None with 0 for non-nullable int columns
        for row in buffer:
            for key, val in row.items():
                if val is None:
                    arrow_field = schema.field(key)
                    if arrow_field.type in (pa.int32(), pa.int64()):
                        row[key] = 0
                    elif arrow_field.type == pa.float64():
                        row[key] = 0.0

        table = pa.Table.from_pylist(buffer, schema=schema)

        # FIX: each flush gets its own file number — never overwrite
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
