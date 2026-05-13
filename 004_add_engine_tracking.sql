-- 004_add_engine_tracking.sql
-- Adds engine tracking columns to all tables for LC0/Stockfish coexistence
-- Run this migration BEFORE processing with LC0

-- Add columns to games table
ALTER TABLE games ADD COLUMN evaluated_by TEXT DEFAULT 'stockfish';
ALTER TABLE games ADD COLUMN evaluator_version TEXT;
ALTER TABLE games ADD COLUMN evaluated_at TIMESTAMP;

-- Add columns to actual_moves table  
ALTER TABLE actual_moves ADD COLUMN evaluated_by TEXT DEFAULT 'stockfish';
ALTER TABLE actual_moves ADD COLUMN evaluator_version TEXT;

-- Add columns to possible_move_evals table
ALTER TABLE possible_move_evals ADD COLUMN evaluated_by TEXT DEFAULT 'stockfish';
ALTER TABLE possible_move_evals ADD COLUMN evaluator_version TEXT;

-- Create indexes for fast filtering
CREATE INDEX IF NOT EXISTS idx_games_engine ON games(evaluated_by, evaluator_version);
CREATE INDEX IF NOT EXISTS idx_actual_moves_engine ON actual_moves(evaluated_by, evaluator_version);
CREATE INDEX IF NOT EXISTS idx_possible_move_evals_engine ON possible_move_evals(evaluated_by, evaluator_version);

-- Create views for easy filtering
DROP VIEW IF EXISTS lc0_games;
CREATE VIEW lc0_games AS SELECT * FROM games WHERE evaluated_by = 'lc0';

DROP VIEW IF EXISTS lc0_moves;
CREATE VIEW lc0_moves AS SELECT * FROM actual_moves WHERE evaluated_by = 'lc0';

DROP VIEW IF EXISTS lc0_possible_moves;
CREATE VIEW lc0_possible_moves AS SELECT * FROM possible_move_evals WHERE evaluated_by = 'lc0';

DROP VIEW IF EXISTS stockfish_games;
CREATE VIEW stockfish_games AS SELECT * FROM games WHERE evaluated_by = 'stockfish' OR evaluated_by IS NULL;

DROP VIEW IF EXISTS stockfish_moves;
CREATE VIEW stockfish_moves AS SELECT * FROM actual_moves WHERE evaluated_by = 'stockfish' OR evaluated_by IS NULL;
