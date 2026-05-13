"""
Parquet Schema with Engine Tracking

This defines the Parquet schema to match SQLite tables with engine annotations.
All three tables include evaluated_by and evaluator_version for filtering.
"""

import pyarrow as pa

# Schema for games table (with engine tracking)
GAMES_SCHEMA = pa.schema([
    ('game_id', pa.string()),
    ('white', pa.string()),
    ('black', pa.string()),
    ('white_elo', pa.int32()),
    ('black_elo', pa.int32()),
    ('result', pa.string()),
    ('event', pa.string()),
    ('site', pa.string()),
    ('game_date', pa.string()),
    ('round', pa.string()),
    ('eco', pa.string()),
    ('time_control', pa.string()),
    ('game_hash', pa.string()),
    # ENGINE TRACKING FIELDS
    ('evaluated_by', pa.string()),           # 'lc0' or 'stockfish'
    ('evaluator_version', pa.string()),      # Network hash or SF version
    ('evaluated_at', pa.string()),           # ISO timestamp
    ('pgn_text', pa.string()),
])

# Schema for actual_moves table (with engine tracking)
# Note: planes_before/after are BLOBs in SQLite, typically excluded from Parquet for size
MOVES_SCHEMA = pa.schema([
    ('game_id', pa.string()),
    ('move_no', pa.int32()),
    ('move_uci', pa.string()),
    ('fen_before', pa.string()),
    ('fen_after', pa.string()),
    ('eval_before', pa.int32()),
    ('eval_after', pa.int32()),
    ('win_prob_white_before', pa.float32()),
    ('win_prob_draw_before', pa.float32()),
    ('win_prob_black_before', pa.float32()),
    ('win_prob_white_after', pa.float32()),
    ('win_prob_draw_after', pa.float32()),
    ('win_prob_black_after', pa.float32()),
    # Planes excluded from Parquet to save space (store separately if needed)
    # ('planes_before', pa.binary()),
    # ('planes_after', pa.binary()),
    ('time_spent_ms', pa.int32()),
    ('time_remaining_ms', pa.int32()),
    # ENGINE TRACKING FIELDS - CRITICAL FOR FILTERING
    ('evaluated_by', pa.string()),           # 'lc0' or 'stockfish'
    ('evaluator_version', pa.string()),      # Network hash or SF version
])

# Schema for possible_move_evals table (with engine tracking)
POSSIBLE_MOVES_SCHEMA = pa.schema([
    ('game_id', pa.string()),
    ('move_no', pa.int32()),
    ('move_uci', pa.string()),
    ('centipawn', pa.int32()),
    ('mate_score', pa.int32()),
    ('win_prob_white', pa.float32()),
    ('win_prob_draw', pa.float32()),
    ('win_prob_black', pa.float32()),
    ('nodes', pa.int64()),
    ('depth', pa.int32()),
    ('pv', pa.string()),
    # ENGINE TRACKING FIELDS - CRITICAL FOR FILTERING
    ('evaluated_by', pa.string()),           # 'lc0' or 'stockfish'
    ('evaluator_version', pa.string()),      # Network hash or SF version
])

# Partitioning strategy for Parquet files
# Partition by engine and version for efficient filtering
PARTITION_COLS = ['evaluated_by', 'evaluator_version']

"""
Example directory structure:
output/
  lc0/
    703810abc123/
      games/
        part-00000.parquet
        part-00001.parquet
      moves/
        part-00000.parquet
      possible_moves/
        part-00000.parquet
  stockfish/
    sf16/
      games/
      moves/
      possible_moves/

Training data loading:
"""
TRAINING_EXAMPLE = """
import pandas as pd
import pyarrow.parquet as pq

# Load only LC0 data (efficient - uses partitioning)
lc0_moves = pd.read_parquet(
    'output/lc0/',
    filters=[('evaluated_by', '=', 'lc0')]
)

# Or load specific version
lc0_v1_moves = pd.read_parquet(
    'output/lc0/703810abc123/moves/'
)

# Combine with Stockfish data if needed
sf_moves = pd.read_parquet('output/stockfish/')
all_moves = pd.concat([lc0_moves, sf_moves])

# Filter in pandas
lc0_only = all_moves[all_moves['evaluated_by'] == 'lc0']
"""

"""
Verification query to ensure SQLite and Parquet match:
"""
VERIFICATION_SQL = """
-- Check that all records have engine annotations
SELECT 
    'games' as table_name,
    COUNT(*) as total,
    COUNT(evaluated_by) as with_engine,
    COUNT(DISTINCT evaluated_by) as distinct_engines
FROM games
UNION ALL
SELECT 
    'actual_moves',
    COUNT(*),
    COUNT(evaluated_by),
    COUNT(DISTINCT evaluated_by)
FROM actual_moves
UNION ALL
SELECT 
    'possible_move_evals',
    COUNT(*),
    COUNT(evaluated_by),
    COUNT(DISTINCT evaluated_by)
FROM possible_move_evals;
"""

print(__doc__)
print("\nSchemas defined:")
print(f"  Games: {len(GAMES_SCHEMA)} fields")
print(f"  Moves: {len(MOVES_SCHEMA)} fields")
print(f"  Possible moves: {len(POSSIBLE_MOVES_SCHEMA)} fields")
print("\nAll schemas include evaluated_by and evaluator_version for filtering.")
