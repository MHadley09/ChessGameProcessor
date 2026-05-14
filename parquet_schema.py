"""
Parquet Schema with Engine Tracking - FIXED

Schemas now match the old SQLite chess_evaluator tables 1:1.
All three tables include evaluated_by and evaluator_version for filtering.
"""

import pyarrow as pa

# Schema for games table — matches old SQLite games table
GAMES_SCHEMA = pa.schema([
    ('game_id', pa.string()),
    ('game_order', pa.int64()),
    ('event', pa.string()),
    ('site', pa.string()),
    ('date_played', pa.string()),
    ('round', pa.string()),
    ('white', pa.string()),           # SHA-256 hashed
    ('black', pa.string()),           # SHA-256 hashed
    ('result', pa.string()),
    ('white_elo', pa.int32()),
    ('white_rating_diff', pa.int32()),
    ('black_elo', pa.int32()),
    ('black_rating_diff', pa.int32()),
    ('white_title', pa.string()),
    ('black_title', pa.string()),
    ('winner', pa.string()),          # SHA-256 hashed (or None for draw)
    ('winner_elo', pa.int32()),
    ('loser', pa.string()),           # SHA-256 hashed (or None for draw)
    ('loser_elo', pa.int32()),
    ('winner_loser_elo_diff', pa.int32()),
    ('eco', pa.string()),
    ('termination', pa.string()),
    ('time_control', pa.string()),
    ('utc_date', pa.string()),
    ('utc_time', pa.string()),
    ('variant', pa.string()),
    ('ply_count', pa.int32()),
    ('game_hash', pa.string()),
    ('evaluated_by', pa.string()),
    ('evaluator_version', pa.string()),
    ('evaluated_at', pa.string()),
    ('pgn_text', pa.string()),
])

# Schema for actual_moves table — matches old SQLite actual_moves table
MOVES_SCHEMA = pa.schema([
    ('game_id', pa.string()),
    ('move_no', pa.int32()),
    ('move_no_pair', pa.int32()),
    ('player', pa.string()),          # SHA-256 hashed
    ('notation', pa.string()),        # SAN
    ('move', pa.string()),            # UCI
    ('from_square', pa.string()),
    ('to_square', pa.string()),
    ('piece', pa.string()),
    ('promotion', pa.string()),
    ('color', pa.string()),
    ('fen_before', pa.string()),
    ('fen_after', pa.string()),
    ('time_remaining', pa.float64()),
    ('time_spent', pa.float64()),
    ('game_to_position', pa.string()),
    ('white_win_perc_before', pa.float64()),
    ('black_win_perc_before', pa.float64()),
    ('draw_perc_before', pa.float64()),
    ('white_win_perc_after', pa.float64()),
    ('black_win_perc_after', pa.float64()),
    ('draw_perc_after', pa.float64()),
    ('static_eval_before', pa.float64()),
    ('static_eval_after', pa.float64()),
    ('eval_before', pa.float64()),
    ('mate_count_before', pa.float64()),
    ('eval_after', pa.float64()),
    ('mate_count_after', pa.float64()),
    # Planes generated at training time from fen_before/fen_after
    ('evaluated_by', pa.string()),
    ('evaluator_version', pa.string()),
])

# Schema for possible_move_evals table — matches old SQLite possible_move_evals table
POSSIBLE_MOVES_SCHEMA = pa.schema([
    ('game_id', pa.string()),
    ('move_no', pa.int32()),
    ('move_no_pair', pa.int32()),
    ('notation', pa.string()),        # SAN
    ('move', pa.string()),            # UCI
    ('from_square', pa.string()),
    ('to_square', pa.string()),
    ('piece', pa.string()),
    ('promotion', pa.string()),
    ('color', pa.string()),
    ('fen_before', pa.string()),
    ('fen_after', pa.string()),
    ('eval', pa.float64()),
    ('mate_count', pa.float64()),
    ('white_win_perc', pa.float64()),
    ('black_win_perc', pa.float64()),
    ('draw_perc', pa.float64()),
    ('nodes', pa.int64()),
    ('depth', pa.int32()),
    ('pv', pa.string()),
    ('evaluated_by', pa.string()),
    ('evaluator_version', pa.string()),
])

PARTITION_COLS = ['evaluated_by', 'evaluator_version']

if __name__ == '__main__':
    print("Schemas defined (matching old SQLite 1:1):")
    print(f"  Games: {len(GAMES_SCHEMA)} fields")
    print(f"  Moves: {len(MOVES_SCHEMA)} fields")
    print(f"  Possible moves: {len(POSSIBLE_MOVES_SCHEMA)} fields")
