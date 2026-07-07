#!/usr/bin/env python3
"""
Puzzle Dataset Processor
========================
Processes Lichess puzzle CSV into V5-compatible NPZ shards with
additional theme and opening classification labels.

Pipeline:
1. Parse lichess_db_puzzle.csv(.zst)
2. Batch-fetch source games from Lichess API (cached in SQLite)
3. Replay each game to the puzzle position
4. Evaluate all legal moves with LC0 (reuses direct evaluator)
5. Write NPZ shards in V5 format + theme/opening fields

Usage:
    python puzzle_dataset_processor.py \\
        --puzzle-csv lichess_db_puzzle.csv \\
        --output dataset/puzzle_v1 \\
        --lc0 /path/to/lc0.exe \\
        --weights large.pb.gz \\
        --backend cuda-fp16 \\
        --lc0-workers 5 \\
        --max-puzzles 100000 \\
        --lichess-token YOUR_TOKEN  # optional, faster fetching
"""

import argparse
import csv
import io
import json
import logging
import os
import sqlite3
import sys
import time
import struct
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import chess
import chess.pgn
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Lazy imports for evaluator (only needed if LC0 is used)
# ---------------------------------------------------------------------------
SyncDirectEvaluator = None

def _lazy_import_evaluator():
    global SyncDirectEvaluator
    if SyncDirectEvaluator is None:
        from direct_evaluator import SyncDirectEvaluator as _SDE
        SyncDirectEvaluator = _SDE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LICHESS_API_BASE = "https://lichess.org"
BATCH_EXPORT_ENDPOINT = "/games/export/_ids"
SINGLE_EXPORT_ENDPOINT = "/game/export/{game_id}"
MAX_IDS_PER_BATCH = 300
RATE_LIMIT_ANONYMOUS = 15  # games/sec
RATE_LIMIT_AUTHENTICATED = 25

# Complete theme label space (62 themes, alphabetically sorted)
ALL_THEMES = sorted([
    "advancedPawn", "anastasiaMate", "arabianMate", "attackingF2F7",
    "attraction", "backRankMate", "bishopEndgame", "bodenMate",
    "capturingDefender", "castling", "clearance", "crushing",
    "defensiveMove", "deflection", "discoveredAttack", "doubleCheck",
    "doubleBishopMate", "dovetailMate", "endgame", "enPassant",
    "equality", "exposedKing", "fork", "hangingPiece",
    "hookMate", "interference", "intermezzo", "killBoxMate",
    "kingsideAttack", "knightEndgame", "long", "master",
    "masterVsMaster", "mate", "mateIn1", "mateIn2",
    "mateIn3", "mateIn4", "mateIn5", "middlegame",
    "oneMove", "opening", "pawnEndgame", "pin",
    "promotion", "queenEndgame", "queenRookEndgame", "queensideAttack",
    "quietMove", "rookEndgame", "sacrifice", "short",
    "skewer", "smotheredMate", "superGM", "trappedPiece",
    "underPromotion", "veryLong", "vukovicMate", "xRayAttack",
    "advantage",  "kingsideAttack",
])
# Deduplicate and re-sort (kingsideAttack was listed twice above)
ALL_THEMES = sorted(list(set(ALL_THEMES)))
THEME_TO_IDX = {t: i for i, t in enumerate(ALL_THEMES)}
NUM_THEMES = len(ALL_THEMES)

# V5-compatible board encoding constants
NUM_PLANES = 23
MAX_POSSIBLE_MOVES = 220

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Game Cache (SQLite)
# ---------------------------------------------------------------------------
class GameCache:
    """SQLite cache for fetched Lichess game PGNs."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                pgn TEXT,
                fetched_at REAL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS failed_fetches (
                game_id TEXT PRIMARY KEY,
                error TEXT,
                failed_at REAL
            )
        """)
        self.conn.commit()

    def get(self, game_id: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT pgn FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        return row[0] if row else None

    def get_many(self, game_ids: List[str]) -> Dict[str, str]:
        """Fetch multiple games from cache. Returns {game_id: pgn}."""
        result = {}
        # SQLite has a limit on query params, chunk if needed
        for i in range(0, len(game_ids), 500):
            chunk = game_ids[i:i+500]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT game_id, pgn FROM games WHERE game_id IN ({placeholders})",
                chunk,
            ).fetchall()
            for gid, pgn in rows:
                result[gid] = pgn
        return result

    def put(self, game_id: str, pgn: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO games (game_id, pgn, fetched_at) VALUES (?, ?, ?)",
            (game_id, pgn, time.time()),
        )

    def put_failed(self, game_id: str, error: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO failed_fetches (game_id, error, failed_at) VALUES (?, ?, ?)",
            (game_id, error, time.time()),
        )

    def is_failed(self, game_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM failed_fetches WHERE game_id = ?", (game_id,)
        ).fetchone()
        return row is not None

    def commit(self):
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Lichess API Client
# ---------------------------------------------------------------------------
class LichessClient:
    """Fetches games from Lichess API with rate limiting and caching."""

    def __init__(self, cache: GameCache, token: Optional[str] = None):
        self.cache = cache
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/x-chess-pgn"
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.rate_limit = RATE_LIMIT_AUTHENTICATED if token else RATE_LIMIT_ANONYMOUS
        self._last_request_time = 0.0
        self._games_since_last_wait = 0

    def _rate_wait(self, n_games: int):
        """Enforce rate limiting based on games returned."""
        now = time.time()
        elapsed = now - self._last_request_time
        min_wait = n_games / self.rate_limit
        if elapsed < min_wait:
            time.sleep(min_wait - elapsed)
        self._last_request_time = time.time()

    def fetch_games_batch(self, game_ids: List[str]) -> Dict[str, str]:
        """Fetch up to 300 games via POST /games/export/_ids.
        
        Returns {game_id: pgn_text} for successfully fetched games.
        """
        # Check cache first
        cached = self.cache.get_many(game_ids)
        uncached = [gid for gid in game_ids
                    if gid not in cached and not self.cache.is_failed(gid)]

        if not uncached:
            return cached

        result = dict(cached)

        # Batch in groups of MAX_IDS_PER_BATCH
        for i in range(0, len(uncached), MAX_IDS_PER_BATCH):
            batch = uncached[i:i + MAX_IDS_PER_BATCH]
            self._rate_wait(len(batch))

            try:
                resp = self.session.post(
                    f"{LICHESS_API_BASE}{BATCH_EXPORT_ENDPOINT}",
                    data=",".join(batch),
                    params={"clocks": "true", "opening": "true", "pgnInJson": "false"},
                    timeout=120,
                    stream=True,
                )
                resp.raise_for_status()

                # Parse PGN stream — games separated by double newline
                pgn_text = resp.text
                games_in_response = self._split_pgn_stream(pgn_text)

                for gid, pgn in games_in_response.items():
                    self.cache.put(gid, pgn)
                    result[gid] = pgn

                # Mark unfound games as failed
                found_ids = set(games_in_response.keys())
                for gid in batch:
                    if gid not in found_ids:
                        self.cache.put_failed(gid, "not_in_response")

                self.cache.commit()

            except requests.RequestException as e:
                log.warning(f"Batch fetch failed: {e}")
                for gid in batch:
                    self.cache.put_failed(gid, str(e))
                self.cache.commit()
                time.sleep(5)  # Back off on error

        return result

    def _split_pgn_stream(self, pgn_text: str) -> Dict[str, str]:
        """Split a multi-game PGN stream into {game_id: pgn} pairs."""
        games = {}
        sio = io.StringIO(pgn_text)
        while True:
            game = chess.pgn.read_game(sio)
            if game is None:
                break
            site = game.headers.get("Site", "")
            # Extract game ID from Site header: https://lichess.org/XXXXX
            gid = site.rstrip("/").split("/")[-1]
            if gid:
                # Reconstruct the PGN text for this game
                exporter = chess.pgn.StringExporter(
                    headers=True, variations=True, comments=True
                )
                pgn_str = game.accept(exporter)
                games[gid] = pgn_str
        return games

    def fetch_single_game(self, game_id: str) -> Optional[str]:
        """Fetch a single game. Falls back to single-game endpoint."""
        cached = self.cache.get(game_id)
        if cached:
            return cached
        if self.cache.is_failed(game_id):
            return None

        self._rate_wait(1)
        try:
            resp = self.session.get(
                f"{LICHESS_API_BASE}{SINGLE_EXPORT_ENDPOINT.format(game_id=game_id)}",
                params={"clocks": "true", "opening": "true"},
                timeout=30,
            )
            if resp.status_code == 404:
                self.cache.put_failed(game_id, "404")
                self.cache.commit()
                return None
            resp.raise_for_status()
            pgn = resp.text
            self.cache.put(game_id, pgn)
            self.cache.commit()
            return pgn
        except requests.RequestException as e:
            log.warning(f"Single fetch failed for {game_id}: {e}")
            self.cache.put_failed(game_id, str(e))
            self.cache.commit()
            return None


# ---------------------------------------------------------------------------
# Puzzle Parser
# ---------------------------------------------------------------------------
def parse_game_url(url: str) -> Tuple[str, Optional[str], Optional[int]]:
    """Parse Lichess game URL into (game_id, color, ply).
    
    Examples:
        https://lichess.org/yyznGmXs/black#34 → ("yyznGmXs", "black", 34)
        https://lichess.org/gyFeQsOE#35 → ("gyFeQsOE", None, 35)
    """
    # Strip trailing whitespace
    url = url.strip()
    
    # Remove base URL
    path = url.replace("https://lichess.org/", "").replace("http://lichess.org/", "")
    
    # Extract ply from fragment
    ply = None
    if "#" in path:
        path, ply_str = path.rsplit("#", 1)
        try:
            ply = int(ply_str)
        except ValueError:
            pass
    
    # Extract game_id and color from path
    parts = path.strip("/").split("/")
    game_id = parts[0]
    color = parts[1] if len(parts) > 1 else None
    
    return game_id, color, ply


def parse_puzzle_row(row: Dict[str, str]) -> Optional[Dict]:
    """Parse a single puzzle CSV row into a structured dict."""
    try:
        puzzle_id = row.get("PuzzleId", "").strip()
        fen = row.get("FEN", "").strip()
        moves_str = row.get("Moves", "").strip()
        themes_str = row.get("Themes", "").strip()
        game_url = row.get("GameUrl", "").strip()
        opening_str = row.get("OpeningTags", "").strip()
        rating = int(row.get("Rating", 0))
        popularity = int(row.get("Popularity", 0))
        nb_plays = int(row.get("NbPlays", 0))

        if not fen or not moves_str or not game_url:
            return None

        moves = moves_str.split()
        if len(moves) < 2:
            return None

        game_id, color, ply = parse_game_url(game_url)
        if not game_id:
            return None

        # Parse themes into multi-hot vector
        themes_list = [t.strip() for t in themes_str.split() if t.strip()]
        theme_vector = np.zeros(NUM_THEMES, dtype=np.float32)
        for t in themes_list:
            if t in THEME_TO_IDX:
                theme_vector[THEME_TO_IDX[t]] = 1.0

        return {
            "puzzle_id": puzzle_id,
            "fen_before_setup": fen,         # FEN before opponent's setup move
            "puzzle_moves": moves,            # [setup_move, solution_move1, ...]
            "themes": themes_list,
            "theme_vector": theme_vector,
            "game_id": game_id,
            "game_color": color,              # which side the puzzle solver plays as
            "game_ply": ply,
            "opening_tags": opening_str,
            "puzzle_rating": rating,
            "popularity": popularity,
            "nb_plays": nb_plays,
        }
    except Exception as e:
        log.debug(f"Failed to parse puzzle row: {e}")
        return None


# ---------------------------------------------------------------------------
# Board Encoding (V5-compatible, 23 planes)
# ---------------------------------------------------------------------------
PIECE_PLANES = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}

def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode a chess.Board into 23 planes of shape (23, 8, 8).
    
    Plane layout (V5-compatible):
      0-5:   White pieces (P, N, B, R, Q, K)
      6-11:  Black pieces (P, N, B, R, Q, K)
      12-15: Last two moves from/to squares (4 planes)
      16:    Side to move (1 = white, 0 = black)
      17-20: Castling rights (WK, WQ, BK, BQ)
      21:    En passant square
      22:    Halfmove clock (normalized)
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    # Piece planes (0-11)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            rank, file = divmod(sq, 8)
            color_offset = 0 if piece.color == chess.WHITE else 6
            plane_idx = color_offset + PIECE_PLANES[piece.piece_type]
            planes[plane_idx, rank, file] = 1.0

    # Last two moves (planes 12-15)
    move_stack = list(board.move_stack)
    for i, move_back in enumerate(range(1, 3)):
        if len(move_stack) >= move_back:
            m = move_stack[-move_back]
            from_rank, from_file = divmod(m.from_square, 8)
            to_rank, to_file = divmod(m.to_square, 8)
            planes[12 + i * 2, from_rank, from_file] = 1.0
            planes[13 + i * 2, to_rank, to_file] = 1.0

    # Side to move (plane 16)
    if board.turn == chess.WHITE:
        planes[16, :, :] = 1.0

    # Castling rights (planes 17-20)
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[17, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[18, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[19, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[20, :, :] = 1.0

    # En passant (plane 21)
    if board.ep_square is not None:
        ep_rank, ep_file = divmod(board.ep_square, 8)
        planes[21, ep_rank, ep_file] = 1.0

    # Halfmove clock (plane 22), normalized to [0, 1]
    planes[22, :, :] = min(board.halfmove_clock / 100.0, 1.0)

    return planes


# ---------------------------------------------------------------------------
# Position Extraction from Game PGN
# ---------------------------------------------------------------------------
def extract_puzzle_position(
    pgn_text: str,
    puzzle: Dict,
) -> Optional[Dict]:
    """Replay a game PGN to the puzzle position and extract data.
    
    Returns dict with:
      - board: chess.Board at puzzle position (after setup move applied)
      - actual_move: the move actually played in the game at this position
      - game_to_position: UCI moves from game start to puzzle position
      - white_elo, black_elo: player ratings
      - time_control: game time control string
      - pgn_truncated: PGN text up to puzzle position
    """
    try:
        sio = io.StringIO(pgn_text)
        game = chess.pgn.read_game(sio)
        if game is None:
            return None

        # Extract metadata
        white_elo = _safe_int(game.headers.get("WhiteElo", "0"))
        black_elo = _safe_int(game.headers.get("BlackElo", "0"))
        time_control = game.headers.get("TimeControl", "")
        result = game.headers.get("Result", "*")

        # Replay to target ply
        target_ply = puzzle.get("game_ply")
        setup_move_uci = puzzle["puzzle_moves"][0]

        board = game.board()
        move_list = []
        node = game

        # Walk through the game move by move
        for i, child_node in enumerate(game.mainline()):
            current_ply = i  # 0-indexed ply count
            move = child_node.move
            move_list.append(move.uci())
            board.push(move)

            # Check if we've reached the puzzle position
            # The puzzle FEN is BEFORE the setup move.
            # After applying setup_move, we're at the puzzle solving position.
            # The target_ply from the URL should correspond to the ply
            # where the puzzle position starts.
            if target_ply is not None and (current_ply + 1) == target_ply:
                # We're at the ply indicated by the URL
                # Verify: current board should have the setup move's result
                break

        # Alternative: match by FEN if ply doesn't work
        # The FEN in the CSV is BEFORE the setup move
        puzzle_fen_before = puzzle["fen_before_setup"]
        
        # Try to find position by FEN match
        board2 = game.board()
        move_list2 = []
        actual_move_in_game = None
        found = False

        for child_node in game.mainline():
            move = child_node.move
            current_fen_parts = board2.fen().split(" ")[:4]
            target_fen_parts = puzzle_fen_before.split(" ")[:4]
            
            if current_fen_parts == target_fen_parts:
                # Found the position BEFORE the setup move
                # The setup move should be this move
                if move.uci() == setup_move_uci:
                    # Apply the setup move
                    board2.push(move)
                    move_list2.append(move.uci())
                    
                    # Now we're at the puzzle position
                    # The next move in the game is the "actual move"
                    # (the puzzle solution IF the player found it)
                    next_node = child_node
                    # Get the next move if it exists
                    if next_node.variations:
                        actual_move_in_game = next_node.variations[0].move.uci()
                    
                    found = True
                    break

            move_list2.append(move.uci())
            board2.push(move)

        if not found:
            # Fallback: try matching just piece positions (ignore turn/castling/ep)
            board3 = game.board()
            move_list3 = []
            for child_node in game.mainline():
                move = child_node.move
                # Compare just piece placement
                current_pieces = board3.fen().split(" ")[0]
                target_pieces = puzzle_fen_before.split(" ")[0]
                
                if current_pieces == target_pieces:
                    if move.uci() == setup_move_uci:
                        board3.push(move)
                        move_list3.append(move.uci())
                        next_node = child_node
                        if next_node.variations:
                            actual_move_in_game = next_node.variations[0].move.uci()
                        found = True
                        board2 = board3
                        move_list2 = move_list3
                        break

                move_list3.append(move.uci())
                board3.push(move)

        if not found:
            log.debug(
                f"Could not find puzzle position in game {puzzle['game_id']} "
                f"for puzzle {puzzle['puzzle_id']}"
            )
            return None

        # Determine result from game perspective
        if result == "1-0":
            result_wdl = [1, 0, 0]  # White won
        elif result == "0-1":
            result_wdl = [0, 0, 1]  # Black won
        else:
            result_wdl = [0, 1, 0]  # Draw

        return {
            "board": board2,
            "actual_move": actual_move_in_game or puzzle["puzzle_moves"][1],
            "game_to_position": " ".join(move_list2),
            "move_history": move_list2,
            "white_elo": white_elo,
            "black_elo": black_elo,
            "time_control": time_control,
            "result_wdl": result_wdl,
        }

    except Exception as e:
        log.debug(f"Error extracting position from game {puzzle.get('game_id')}: {e}")
        return None


def _safe_int(s: str) -> int:
    try:
        return int(s.strip().rstrip("?"))
    except (ValueError, AttributeError):
        return 0


# ---------------------------------------------------------------------------
# LC0 Evaluation (reuses direct_evaluator infrastructure)
# ---------------------------------------------------------------------------
def evaluate_position(
    board: chess.Board,
    evaluator,
) -> Optional[Dict]:
    """Evaluate all legal moves at a position using the evaluator.
    
    Returns dict with:
      - possible_uci: list of UCI move strings
      - possible_scalars: (M, 13) array of per-move features
      - possible_mask: (M,) binary mask
      - wdl: [W, D, L] from engine perspective
    """
    try:
        results = evaluator.evaluate_all_legal_moves(board)
        if not results:
            return None

        legal_moves = list(board.legal_moves)
        n_moves = len(legal_moves)
        
        possible_uci = []
        possible_scalars = np.zeros((MAX_POSSIBLE_MOVES, 13), dtype=np.float32)
        possible_mask = np.zeros(MAX_POSSIBLE_MOVES, dtype=np.float32)
        
        # Sort results to match legal moves order
        result_by_uci = {r["move"]: r for r in results}
        
        for i, move in enumerate(legal_moves):
            if i >= MAX_POSSIBLE_MOVES:
                break
            uci = move.uci()
            possible_uci.append(uci)
            possible_mask[i] = 1.0
            
            if uci in result_by_uci:
                r = result_by_uci[uci]
                # Populate per-move scalars (V5 format)
                # [0]: score_cp, [1-3]: wdl (W,D,L), [4]: policy, 
                # [5]: nodes, [6]: depth, [7]: is_capture, [8]: is_check,
                # [9]: piece_type, [10]: from_sq, [11]: to_sq, [12]: is_excellent
                possible_scalars[i, 0] = r.get("score_cp", 0)
                wdl = r.get("wdl", [0, 0, 0])
                possible_scalars[i, 1] = wdl[0] if len(wdl) > 0 else 0
                possible_scalars[i, 2] = wdl[1] if len(wdl) > 1 else 0
                possible_scalars[i, 3] = wdl[2] if len(wdl) > 2 else 0
                possible_scalars[i, 4] = r.get("policy", 0)
                possible_scalars[i, 5] = r.get("nodes", 0)
                possible_scalars[i, 6] = r.get("depth", 0)
                # Board-derived features
                possible_scalars[i, 7] = 1.0 if board.is_capture(move) else 0.0
                possible_scalars[i, 8] = 1.0 if board.gives_check(move) else 0.0
                possible_scalars[i, 9] = float(board.piece_type_at(move.from_square) or 0)
                possible_scalars[i, 10] = float(move.from_square)
                possible_scalars[i, 11] = float(move.to_square)
                possible_scalars[i, 12] = 0.0  # is_excellent (computed later)
        
        # Pad UCI list
        while len(possible_uci) < MAX_POSSIBLE_MOVES:
            possible_uci.append("")

        # Overall position WDL from best move
        best_result = results[0] if results else {}
        position_wdl = best_result.get("wdl", [33, 34, 33])

        return {
            "possible_uci": possible_uci,
            "possible_scalars": possible_scalars,
            "possible_mask": possible_mask,
            "position_wdl": position_wdl,
        }

    except Exception as e:
        log.debug(f"Evaluation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Tabular Feature Construction (V5-compatible, 20-dim)
# ---------------------------------------------------------------------------
def build_tabular(
    board: chess.Board,
    white_elo: int,
    black_elo: int,
    time_control: str,
    result_wdl: List[int],
    actual_idx: int,
    puzzle_rating: int,
    move_number: int,
) -> np.ndarray:
    """Build 20-dim tabular feature vector (V5 format).
    
    Slots:
      0: white_elo (normalized)
      1: black_elo (normalized)
      2: elo_diff (white - black, normalized)
      3: time_initial (seconds, normalized)
      4: time_increment (seconds, normalized)
      5: move_number (normalized)
      6: side_to_move (0=white, 1=black)
      7: result_w
      8: result_d
      9: result_l
      10: material_balance (centipawns, normalized)
      11: num_pieces
      12: is_check
      13: castling_rights_count
      14: pawn_structure_score (placeholder)
      15: king_safety_score (placeholder)
      16: center_control (placeholder)
      17: puzzle_rating (normalized) — replaces frac_mistake in puzzle context
      18-19: reserved (frac_mistake_moves / frac_excellent_moves in V5)
    """
    tab = np.zeros(20, dtype=np.float32)

    # Elo features
    tab[0] = white_elo / 3000.0
    tab[1] = black_elo / 3000.0
    tab[2] = (white_elo - black_elo) / 1000.0

    # Time control
    tc_parts = time_control.split("+")
    try:
        tab[3] = float(tc_parts[0]) / 600.0 if tc_parts else 0
        tab[4] = float(tc_parts[1]) / 30.0 if len(tc_parts) > 1 else 0
    except (ValueError, IndexError):
        pass

    # Move number
    tab[5] = move_number / 80.0

    # Side to move
    tab[6] = 0.0 if board.turn == chess.WHITE else 1.0

    # Result
    tab[7] = float(result_wdl[0])
    tab[8] = float(result_wdl[1])
    tab[9] = float(result_wdl[2])

    # Material balance
    piece_values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
    white_mat = sum(piece_values.get(p.piece_type, 0)
                    for p in board.piece_map().values() if p.color == chess.WHITE)
    black_mat = sum(piece_values.get(p.piece_type, 0)
                    for p in board.piece_map().values() if p.color == chess.BLACK)
    tab[10] = (white_mat - black_mat) / 2000.0

    # Num pieces
    tab[11] = len(board.piece_map()) / 32.0

    # Is check
    tab[12] = 1.0 if board.is_check() else 0.0

    # Castling rights count
    cr = bin(board.castling_rights).count("1")
    tab[13] = cr / 4.0

    # Puzzle rating (replaces V5's frac_mistake slot 17)
    tab[17] = puzzle_rating / 3000.0

    return tab


# ---------------------------------------------------------------------------
# Shard Writer
# ---------------------------------------------------------------------------
class ShardWriter:
    """Writes V5-compatible NPZ shards with puzzle-specific extensions."""

    def __init__(self, output_dir: str, shard_size: int = 5000, prefix: str = "puzzle"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.prefix = prefix
        self.shard_idx = 0
        self.buffer = []

    def add(self, example: Dict):
        self.buffer.append(example)
        if len(self.buffer) >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return

        n = len(self.buffer)
        shard_path = self.output_dir / f"{self.prefix}_{self.shard_idx:06d}.npz"

        # Aggregate arrays
        data = {
            "current_planes": np.stack([e["current_planes"] for e in self.buffer]),
            "possible_scalars": np.stack([e["possible_scalars"] for e in self.buffer]),
            "possible_mask": np.stack([e["possible_mask"] for e in self.buffer]),
            "tabular": np.stack([e["tabular"] for e in self.buffer]),
            "actual_idx": np.array([e["actual_idx"] for e in self.buffer], dtype=np.int64),
            "is_mistake": np.array([e["is_mistake"] for e in self.buffer], dtype=np.float32),
            "win_prob_before": np.stack([e["win_prob_before"] for e in self.buffer]),
            # Puzzle-specific extensions
            "themes": np.stack([e["themes"] for e in self.buffer]),
            "puzzle_rating": np.array([e["puzzle_rating"] for e in self.buffer], dtype=np.int32),
        }

        # String arrays stored as object arrays
        data["fen_before"] = np.array([e["fen_before"] for e in self.buffer], dtype=object)
        data["puzzle_id"] = np.array([e["puzzle_id"] for e in self.buffer], dtype=object)
        data["opening_tags"] = np.array([e["opening_tags"] for e in self.buffer], dtype=object)
        data["game_to_position"] = np.array(
            [e["game_to_position"] for e in self.buffer], dtype=object
        )

        # Opening index for classification
        data["opening_idx"] = np.array(
            [e.get("opening_idx", -1) for e in self.buffer], dtype=np.int32
        )

        np.savez_compressed(str(shard_path), **data)
        log.info(f"Wrote shard {shard_path.name} ({n} examples)")

        self.shard_idx += 1
        self.buffer = []

    def finalize(self):
        self._flush()
        log.info(f"Finalized: {self.shard_idx} shards written to {self.output_dir}")


# ---------------------------------------------------------------------------
# Opening Label Builder
# ---------------------------------------------------------------------------
class OpeningLabelBuilder:
    """Builds a label mapping from opening tag strings to integer indices."""

    def __init__(self):
        self.tag_to_idx: Dict[str, int] = {}
        self.idx_to_tag: Dict[int, str] = {}
        self._counter = 0

    def get_or_create(self, tag: str) -> int:
        if not tag:
            return -1
        if tag not in self.tag_to_idx:
            self.tag_to_idx[tag] = self._counter
            self.idx_to_tag[self._counter] = tag
            self._counter += 1
        return self.tag_to_idx[tag]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"tag_to_idx": self.tag_to_idx, "idx_to_tag": self.idx_to_tag}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "OpeningLabelBuilder":
        builder = cls()
        with open(path) as f:
            data = json.load(f)
        builder.tag_to_idx = data["tag_to_idx"]
        builder.idx_to_tag = {int(k): v for k, v in data["idx_to_tag"].items()}
        builder._counter = max(builder.idx_to_tag.keys()) + 1 if builder.idx_to_tag else 0
        return builder

    def num_classes(self) -> int:
        return self._counter


# ---------------------------------------------------------------------------
# Main Processing Pipeline
# ---------------------------------------------------------------------------
def process_puzzles(args):
    """Main processing pipeline."""
    log.info(f"Starting puzzle dataset processor")
    log.info(f"Puzzle CSV: {args.puzzle_csv}")
    log.info(f"Output dir: {args.output}")
    log.info(f"Max puzzles: {args.max_puzzles}")

    # Initialize game cache
    cache_db = os.path.join(args.output, "game_cache.db")
    os.makedirs(args.output, exist_ok=True)
    cache = GameCache(cache_db)
    log.info(f"Game cache: {cache_db} ({cache.count()} cached games)")

    # Initialize Lichess client
    client = LichessClient(cache, token=args.lichess_token)

    # Initialize LC0 evaluator if requested
    evaluator = None
    if args.lc0:
        _lazy_import_evaluator()
        evaluator = SyncDirectEvaluator(
            lc0_path=args.lc0,
            weights_path=args.weights,
            backend=args.backend,
            batch_size=args.lc0_batch,
            nodes_mult=args.lc0_nodes_mult,
            max_nodes=args.lc0_max_nodes,
        )
        log.info(f"LC0 evaluator initialized: {args.lc0}")

    # Initialize shard writer
    writer = ShardWriter(
        output_dir=os.path.join(args.output, "shards"),
        shard_size=args.shard_size,
        prefix="puzzle",
    )

    # Initialize opening label builder
    opening_builder = OpeningLabelBuilder()

    # Parse puzzle CSV
    log.info("Parsing puzzle CSV...")
    puzzles = []
    game_ids_needed: Set[str] = set()

    # Handle .zst compression
    csv_path = args.puzzle_csv
    if csv_path.endswith(".zst"):
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        with open(csv_path, "rb") as f:
            reader_stream = dctx.stream_reader(f)
            text_stream = io.TextIOWrapper(reader_stream, encoding="utf-8")
            csv_reader = csv.DictReader(text_stream)
            for i, row in enumerate(csv_reader):
                if args.max_puzzles and i >= args.max_puzzles:
                    break
                puzzle = parse_puzzle_row(row)
                if puzzle:
                    puzzles.append(puzzle)
                    game_ids_needed.add(puzzle["game_id"])
                if (i + 1) % 100000 == 0:
                    log.info(f"  Parsed {i+1} rows, {len(puzzles)} valid puzzles")
    else:
        with open(csv_path, "r", encoding="utf-8") as f:
            csv_reader = csv.DictReader(f)
            for i, row in enumerate(csv_reader):
                if args.max_puzzles and i >= args.max_puzzles:
                    break
                puzzle = parse_puzzle_row(row)
                if puzzle:
                    puzzles.append(puzzle)
                    game_ids_needed.add(puzzle["game_id"])
                if (i + 1) % 100000 == 0:
                    log.info(f"  Parsed {i+1} rows, {len(puzzles)} valid puzzles")

    log.info(f"Parsed {len(puzzles)} puzzles from {len(game_ids_needed)} unique games")

    # Phase 2: Fetch games in batches
    game_id_list = sorted(game_ids_needed)
    log.info(f"Fetching {len(game_id_list)} games from Lichess API...")

    fetched = 0
    for batch_start in range(0, len(game_id_list), MAX_IDS_PER_BATCH):
        batch = game_id_list[batch_start:batch_start + MAX_IDS_PER_BATCH]
        result = client.fetch_games_batch(batch)
        fetched += len(result)
        if (batch_start + MAX_IDS_PER_BATCH) % 3000 == 0 or batch_start == 0:
            log.info(f"  Fetched {fetched}/{len(game_id_list)} games")

    log.info(f"Game fetch complete. {cache.count()} games in cache.")

    # Phase 3-5: Process each puzzle
    processed = 0
    skipped = 0
    errors = 0

    for i, puzzle in enumerate(puzzles):
        try:
            # Get game PGN from cache
            pgn_text = cache.get(puzzle["game_id"])
            if not pgn_text:
                skipped += 1
                continue

            # Extract position from game
            position_data = extract_puzzle_position(pgn_text, puzzle)
            if not position_data:
                skipped += 1
                continue

            board = position_data["board"]
            actual_move_uci = position_data["actual_move"]

            # Encode board
            current_planes = board_to_planes(board)

            # Find actual move index among legal moves
            legal_moves = [m.uci() for m in board.legal_moves]
            actual_idx = -1
            for j, m in enumerate(legal_moves):
                if m == actual_move_uci:
                    actual_idx = j
                    break

            if actual_idx == -1:
                # Actual move not in legal moves (shouldn't happen)
                skipped += 1
                continue

            # Evaluate position with LC0 (if available)
            eval_data = None
            if evaluator:
                eval_data = evaluate_position(board, evaluator)

            # Build possible_scalars and possible_mask
            if eval_data:
                possible_scalars = eval_data["possible_scalars"]
                possible_mask = eval_data["possible_mask"]
                possible_uci = eval_data["possible_uci"]
                position_wdl = eval_data["position_wdl"]
                # Re-find actual_idx in evaluator's move order
                for j, uci in enumerate(possible_uci):
                    if uci == actual_move_uci:
                        actual_idx = j
                        break
            else:
                # Without LC0, create basic possible_scalars from legal moves
                n_legal = min(len(legal_moves), MAX_POSSIBLE_MOVES)
                possible_scalars = np.zeros((MAX_POSSIBLE_MOVES, 13), dtype=np.float32)
                possible_mask = np.zeros(MAX_POSSIBLE_MOVES, dtype=np.float32)
                for j in range(n_legal):
                    possible_mask[j] = 1.0
                    move = list(board.legal_moves)[j]
                    possible_scalars[j, 7] = 1.0 if board.is_capture(move) else 0.0
                    possible_scalars[j, 8] = 1.0 if board.gives_check(move) else 0.0
                    possible_scalars[j, 9] = float(board.piece_type_at(move.from_square) or 0)
                    possible_scalars[j, 10] = float(move.from_square)
                    possible_scalars[j, 11] = float(move.to_square)
                position_wdl = [33, 34, 33]

            # Build tabular features
            move_number = len(position_data["move_history"]) // 2
            tabular = build_tabular(
                board=board,
                white_elo=position_data["white_elo"],
                black_elo=position_data["black_elo"],
                time_control=position_data["time_control"],
                result_wdl=position_data["result_wdl"],
                actual_idx=actual_idx,
                puzzle_rating=puzzle["puzzle_rating"],
                move_number=move_number,
            )

            # Compute is_mistake (based on eval drop if LC0 available)
            is_mistake = 0.0
            if eval_data and actual_idx >= 0:
                best_score = possible_scalars[0, 0]  # best move score
                actual_score = possible_scalars[actual_idx, 0]
                ev_drop = abs(best_score - actual_score) / 100.0  # in pawns
                if ev_drop > 0.25:
                    is_mistake = 1.0

            # Opening label
            opening_idx = opening_builder.get_or_create(puzzle["opening_tags"])

            # Win probability before (from position WDL)
            wdl_arr = np.array(position_wdl, dtype=np.float32)
            wdl_sum = wdl_arr.sum()
            if wdl_sum > 0:
                wdl_arr = wdl_arr / wdl_sum
            win_prob_before = wdl_arr

            # Build example
            example = {
                "current_planes": current_planes,
                "possible_scalars": possible_scalars,
                "possible_mask": possible_mask,
                "tabular": tabular,
                "actual_idx": actual_idx,
                "is_mistake": is_mistake,
                "win_prob_before": win_prob_before,
                "fen_before": board.fen(),
                "game_to_position": position_data["game_to_position"],
                "puzzle_id": puzzle["puzzle_id"],
                "themes": puzzle["theme_vector"],
                "opening_tags": puzzle["opening_tags"],
                "opening_idx": opening_idx,
                "puzzle_rating": puzzle["puzzle_rating"],
            }

            writer.add(example)
            processed += 1

            if processed % 1000 == 0:
                log.info(
                    f"Processed {processed}/{len(puzzles)} puzzles "
                    f"(skipped={skipped}, errors={errors})"
                )

        except MemoryError:
            log.warning(f"MemoryError on puzzle {puzzle.get('puzzle_id')}, skipping")
            errors += 1
            continue
        except Exception as e:
            log.debug(f"Error processing puzzle {puzzle.get('puzzle_id')}: {e}")
            errors += 1
            continue

    # Finalize
    writer.finalize()

    # Save opening label mapping
    opening_path = os.path.join(args.output, "opening_labels.json")
    opening_builder.save(opening_path)
    log.info(f"Opening labels saved: {opening_builder.num_classes()} classes")

    # Save theme label mapping
    theme_path = os.path.join(args.output, "theme_labels.json")
    with open(theme_path, "w") as f:
        json.dump({"themes": ALL_THEMES, "theme_to_idx": THEME_TO_IDX}, f, indent=2)

    # Save metadata
    meta_path = os.path.join(args.output, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "total_puzzles_parsed": len(puzzles),
            "total_processed": processed,
            "total_skipped": skipped,
            "total_errors": errors,
            "num_themes": NUM_THEMES,
            "num_openings": opening_builder.num_classes(),
            "shard_size": args.shard_size,
            "lc0_used": args.lc0 is not None,
        }, f, indent=2)

    log.info(f"\nDone! Processed {processed} puzzles, skipped {skipped}, errors {errors}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Process Lichess puzzles into V5-compatible training shards"
    )
    parser.add_argument("--puzzle-csv", required=True,
                        help="Path to lichess_db_puzzle.csv or .csv.zst")
    parser.add_argument("--output", required=True,
                        help="Output directory for shards and metadata")
    parser.add_argument("--max-puzzles", type=int, default=None,
                        help="Max puzzles to process (None = all)")
    parser.add_argument("--shard-size", type=int, default=5000,
                        help="Examples per NPZ shard")

    # Lichess API
    parser.add_argument("--lichess-token", default=None,
                        help="Lichess API token for faster fetching")

    # LC0 (optional — without it, no engine evals, just board features)
    parser.add_argument("--lc0", default=None, help="Path to LC0 executable")
    parser.add_argument("--weights", default=None, help="LC0 weights file")
    parser.add_argument("--backend", default="cuda-fp16", help="LC0 backend")
    parser.add_argument("--lc0-batch", type=int, default=128, help="LC0 batch size")
    parser.add_argument("--lc0-nodes-mult", type=float, default=1.0)
    parser.add_argument("--lc0-max-nodes", type=int, default=218)

    # Filtering
    parser.add_argument("--min-rating", type=int, default=0,
                        help="Minimum puzzle rating to include")
    parser.add_argument("--min-popularity", type=int, default=-100,
                        help="Minimum puzzle popularity to include")
    parser.add_argument("--min-plays", type=int, default=0,
                        help="Minimum play count to include")

    args = parser.parse_args()
    process_puzzles(args)


if __name__ == "__main__":
    main()
