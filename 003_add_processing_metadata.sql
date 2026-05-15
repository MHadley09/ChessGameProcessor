-- 003_add_processing_metadata.sql (CORRECTED FOR SQLITE)
-- Adds tables for deduplication and position caching
-- SQLite compatible - no non-constant defaults on ALTER TABLE

-- Track which files have been processed
CREATE TABLE IF NOT EXISTS processing_log (
    file_path TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    games_processed INTEGER DEFAULT 0,
    games_skipped INTEGER DEFAULT 0,
    positions_evaluated INTEGER DEFAULT 0,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    engine TEXT,
    engine_version TEXT,
    status TEXT CHECK(status IN ('processing', 'completed', 'failed')),
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_processing_log_status ON processing_log(status, completed_at);

-- Cache evaluations to avoid re-computing positions
CREATE TABLE IF NOT EXISTS positions_cache (
    position_hash TEXT NOT NULL,
    fen TEXT NOT NULL,
    evaluated_by TEXT NOT NULL,
    evaluator_version TEXT NOT NULL,
    eval_centipawn INTEGER,
    win_prob_white REAL,
    win_prob_draw REAL,
    win_prob_black REAL,
    mate_score INTEGER,
    nodes INTEGER,
    depth INTEGER,
    pv TEXT,
    cached_at TIMESTAMP,
    PRIMARY KEY (position_hash, evaluated_by, evaluator_version)
);

CREATE INDEX IF NOT EXISTS idx_positions_cache_lookup ON positions_cache(position_hash, evaluated_by, evaluator_version);
CREATE INDEX IF NOT EXISTS idx_positions_cache_fen ON positions_cache(fen);

-- Add hash column to games table for deduplication
ALTER TABLE games ADD COLUMN game_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_games_hash ON games(game_hash);

-- Add dedup timestamp (SQLite allows CURRENT_TIMESTAMP in CREATE TABLE but not ALTER)
ALTER TABLE games ADD COLUMN dedup_processed_at TIMESTAMP;

-- Add processing metadata to games
ALTER TABLE games ADD COLUMN processed_by_version TEXT;
ALTER TABLE games ADD COLUMN processing_time_ms INTEGER;

-- Create indexes for the new columns
CREATE INDEX IF NOT EXISTS idx_games_processed_at ON games(dedup_processed_at);
CREATE INDEX IF NOT EXISTS idx_games_hash_lookup ON games(game_hash);