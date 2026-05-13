#!/usr/bin/env python3
"""
lc0_processor_with_parquet.py

FIXED VERSION - Parquet output now matches 1:1 with old SQLite chess_evaluator fields.
Headers still go to SQLite only. Moves + possible_moves go to Parquet with full parity.

Changes from previous version:
- Games parquet: added hashed white/black, elo diffs, winner/loser fields, termination,
  variant, ply_count, utc_date, utc_time, white_title, black_title, game_order
- Actual moves parquet: added move_no_pair, player (hashed), notation (san), from_square,
  to_square, piece, promotion, color, game_to_position, static_eval_before/after,
  mate_count_before/after. Fixed fen_before bug (was captured after board.push).
  Fixed WDL _after values (were copy-pasted from _before).
- Possible moves parquet: added move_no_pair, notation, from_square, to_square, piece,
  promotion, color, fen_before, fen_after. Renamed fields to match old schema.
"""

import sqlite3
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import chess.pgn
import chess

from lc0_evaluator import LC0Evaluator
from plane_codec import board_to_planes, pack_planes
from deduplication_helpers import GameDeduplicator

try:
    from parquet_writer import ParquetWriter, GameRecord, MoveRecord, PossibleMoveRecord
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("Warning: parquet_writer not available. Install pyarrow: pip install pyarrow")

try:
    from batch_evaluator import SyncBatchEvaluator
    BATCH_EVAL_AVAILABLE = True
except ImportError:
    BATCH_EVAL_AVAILABLE = False


class LC0GameProcessorWithParquet:
    """
    LC0 processor that writes to BOTH SQLite and Parquet.
    Parquet schemas match the old chess_evaluator SQLite tables 1:1.
    """

    def __init__(self,
                 db_path: str,
                 weights_path: str,
                 output_dir: str = "output",
                 engine_path: str = None,
                 backend: str = "cuda-fp16",
                 write_parquet: bool = True,
                 write_sqlite: bool = True,
                 headers_only_sqlite: bool = True,
                 num_engines: int = 16,
                 verbose: bool = True):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.write_parquet = write_parquet and PARQUET_AVAILABLE
        self.write_sqlite = write_sqlite
        self.headers_only_sqlite = headers_only_sqlite
        self.verbose = verbose
        self.engine_path = engine_path

        if write_parquet and not PARQUET_AVAILABLE:
            raise ImportError("Parquet writing requested but parquet_writer not available.")

        print(f"Initializing LC0 processor with {backend} backend...")
        if BATCH_EVAL_AVAILABLE and num_engines > 1:
            print(f"  Using batch evaluator with {num_engines} async engines")
            eval_kwargs = {'weights_path': weights_path, 'backend': backend, 'num_engines': num_engines}
            if engine_path:
                eval_kwargs['engine_path'] = engine_path
            self.evaluator = SyncBatchEvaluator(**eval_kwargs)
            self._use_batch = True
        else:
            eval_kwargs = {'weights_path': weights_path, 'backend': backend, 'threads': 4}
            if engine_path:
                eval_kwargs['engine_path'] = engine_path
            self.evaluator = LC0Evaluator(**eval_kwargs)
            self._use_batch = False

        if db_path:
            self.deduplicator = GameDeduplicator(db_path)
        else:
            self.deduplicator = None

        self.engine_info = {
            'engine': 'lc0',
            'version': self.evaluator.network_info['weights_hash'],
            'backend': backend,
            'weights_file': Path(weights_path).name,
            'evaluated_at': datetime.now().isoformat(),
        }

        self.parquet_writers = {}
        if self.write_parquet:
            engine_dir = self.output_dir / 'lc0' / self.engine_info['version']
            engine_dir.mkdir(parents=True, exist_ok=True)
            self.parquet_writers = {
                'games': ParquetWriter(
                    output_dir=str(engine_dir / 'games'),
                    schema_type='games',
                    max_rows_per_file=10000
                ),
                'moves': ParquetWriter(
                    output_dir=str(engine_dir / 'moves'),
                    schema_type='moves',
                    max_rows_per_file=100000
                ),
                'possible_moves': ParquetWriter(
                    output_dir=str(engine_dir / 'possible_moves'),
                    schema_type='possible_moves',
                    max_rows_per_file=500000
                ),
            }
            print(f"Parquet output: {engine_dir}")

        print(f"Processor ready: lc0-{self.engine_info['version']}")
        print(f"  SQLite: {'enabled' if self.write_sqlite else 'disabled'} ({db_path})")
        print(f"  Parquet: {'enabled' if self.write_parquet else 'disabled'}")

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _hash_player(name: str) -> str:
        return hashlib.sha256(name.encode('utf-8')).hexdigest()

    @staticmethod
    def _safe_elo(val, default=1600) -> int:
        if val and val.isdigit():
            return int(val)
        return default

    @staticmethod
    def _safe_int(val) -> Optional[int]:
        try:
            return int(val) if val and val != '?' else None
        except Exception:
            return None

    def _game_hash(self, game: chess.pgn.Game) -> str:
        headers = game.headers
        moves = ''.join([m.uci() for m in game.mainline_moves()])
        content = f"{headers.get('White', '')}|{headers.get('Black', '')}|{headers.get('Date', '')}|{moves}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def _get_time_spent(node) -> Optional[float]:
        """Return time spent in seconds (matching old emt() call)."""
        try:
            return node.emt() or None
        except Exception:
            return None

    @staticmethod
    def _get_time_remaining(node) -> Optional[float]:
        """Return clock remaining in seconds (matching old clock() call)."""
        try:
            return node.clock() or None
        except Exception:
            return None

    @staticmethod
    def _parse_eval(eval_dict):
        """Parse LC0 eval into (centipawn, mate_count) matching old format."""
        if eval_dict is None:
            return (None, None)
        ev = eval_dict.get('ev')
        # LC0 doesn't give discrete mate counts the way stockfish does,
        # but we mirror the old schema: ev goes in eval, mate in mate_count.
        # A huge |ev| (>=9000) is treated as forced mate for compatibility.
        if ev is not None and abs(ev) >= 9000:
            mate_sign = 1 if ev > 0 else -1
            return (None, mate_sign)  # mate_count = direction
        return (ev, None)

    # ── game header (parquet + sqlite) ───────────────────────────────────

    def _build_game_record(self, game: chess.pgn.Game, game_id: str, game_order: int) -> dict:
        """Build a game record matching the old SQLite games table 1:1."""
        h = game.headers

        white_name = h.get("White", "Unknown")
        black_name = h.get("Black", "Unknown")
        white = self._hash_player(white_name)
        black = self._hash_player(black_name)

        white_elo = self._safe_elo(h.get("WhiteElo"))
        black_elo = self._safe_elo(h.get("BlackElo"))
        result = h.get("Result", "Unknown")

        white_elo_diff = white_elo - black_elo
        black_elo_diff = black_elo - white_elo

        winner = "Unknown"
        loser = "Unknown"
        winner_elo = None
        loser_elo = None
        winner_loser_elo_diff = None

        if result == "1-0":
            winner = white
            loser = black
            winner_elo = white_elo
            loser_elo = black_elo
            winner_loser_elo_diff = white_elo_diff
        elif result == "0-1":
            winner = black
            loser = white
            winner_elo = black_elo
            loser_elo = white_elo
            winner_loser_elo_diff = black_elo_diff
        elif result in ("1/2-1/2", "0.5-0.5"):
            winner = None
            loser = None

        return {
            'game_id': game_id,
            'game_order': game_order,
            'event': h.get("Event", "Unknown"),
            'site': h.get("Site", "Unknown"),
            'date_played': h.get("Date", "Unknown"),
            'round': h.get("Round", ""),
            'white': white,
            'black': black,
            'result': result,
            'white_elo': white_elo,
            'white_rating_diff': white_elo_diff,
            'black_elo': black_elo,
            'black_rating_diff': black_elo_diff,
            'white_title': h.get("WhiteTitle", ""),
            'black_title': h.get("BlackTitle", ""),
            'winner': winner,
            'winner_elo': winner_elo,
            'loser': loser,
            'loser_elo': loser_elo,
            'winner_loser_elo_diff': winner_loser_elo_diff,
            'eco': h.get("ECO") or "-",
            'termination': h.get("Termination") or "Unknown",
            'time_control': h.get("TimeControl") or "Unknown",
            'utc_date': h.get("UTCDate", h.get("Date", "Unknown")),
            'utc_time': h.get("UTCTime", h.get("EndTime", "Unknown")),
            'variant': h.get("Variant") or "Standard",
            'ply_count': len(list(game.mainline_moves())),
            'game_hash': game_id,
            'evaluated_by': 'lc0',
            'evaluator_version': self.engine_info['version'],
            'evaluated_at': self.engine_info['evaluated_at'],
            'pgn_text': str(game)[:10000],
        }

    # ── possible move (mirrors get_possible_move from chess_evaluator) ──

    def _build_possible_move(self, game_id, board, legal_move, fen_before, eval_result):
        """Build one possible-move record matching old possible_move_evals schema."""
        san = board.san(legal_move)
        uci = legal_move.uci()
        from_square = chess.square_name(legal_move.from_square)
        to_square = chess.square_name(legal_move.to_square)
        promotion = None if legal_move.promotion is None else chess.Piece(legal_move.promotion, board.turn).symbol()
        piece = board.piece_at(chess.parse_square(from_square)).symbol()
        color = "White" if board.turn == chess.WHITE else "Black"

        board_copy = board.copy()
        board_copy.push(legal_move)
        move_no_pair = board_copy.fullmove_number
        move_no = board_copy.ply()
        fen_after = board_copy.fen()

        (ev, mate_count) = self._parse_eval(eval_result)

        wdl = eval_result.get('wdl', [333, 334, 333]) if eval_result else [333, 334, 333]
        wdl_total = 1000.0

        return {
            'game_id': game_id,
            'move_no': move_no,
            'move_no_pair': move_no_pair,
            'notation': san,
            'move': uci,
            'from_square': from_square,
            'to_square': to_square,
            'piece': piece,
            'promotion': promotion,
            'color': color,
            'fen_before': fen_before,
            'fen_after': fen_after,
            'eval': ev,
            'mate_count': mate_count,
            'white_win_perc': wdl[0] / wdl_total,
            'black_win_perc': wdl[2] / wdl_total,
            'draw_perc': wdl[1] / wdl_total,
            'evaluated_by': 'lc0',
            'evaluator_version': self.engine_info['version'],
        }

    # ── process single game ─────────────────────────────────────────────

    def _process_game(self, conn, game: chess.pgn.Game, game_id: str, game_order: int) -> Dict:
        headers = game.headers

        # ── game record ──
        game_data = self._build_game_record(game, game_id, game_order)

        # Games/headers always go to SQLite
        if conn:
            # Use a subset of fields that match the SQLite games table
            header_cols = ['game_id', 'game_order', 'event', 'site', 'date_played', 'round',
                          'white', 'black', 'result', 'white_elo', 'white_rating_diff',
                          'black_elo', 'black_rating_diff', 'white_title', 'black_title',
                          'winner', 'winner_elo', 'loser', 'loser_elo', 'winner_loser_elo_diff',
                          'eco', 'termination', 'time_control', 'utc_date', 'utc_time',
                          'variant', 'ply_count', 'game_hash', 'evaluated_by',
                          'evaluator_version', 'evaluated_at']
            header_data = {k: game_data[k] for k in header_cols if k in game_data}
            cols = ', '.join(header_data.keys())
            placeholders = ', '.join(f':{k}' for k in header_data.keys())
            conn.execute(f"INSERT OR REPLACE INTO games ({cols}) VALUES ({placeholders})", header_data)

        # Games also go to parquet with full data
        if self.write_parquet:
            self.parquet_writers['games'].add_game(GameRecord(**game_data))

        # ── iterate moves — two-phase: collect all positions, then batch eval ──
        board = game.board()
        positions = 0
        possible_moves_count = 0
        moves_list = []

        # Phase 1: Walk the game tree, collect all boards to evaluate
        move_metadata = []  # per-move metadata collected during walk
        all_boards = []     # flat list of boards to batch-eval
        board_map = []      # (move_idx, role, pm_idx) to map results back

        # eval_before for move 0 = starting position
        # eval_after for move N = eval_before for move N+1
        # So we need: position_before[0], possible_moves[0], position_before[1], possible_moves[1], ..., position_after[last]
        # Which simplifies to: position[0], pm[0], position[1], pm[1], ..., position[N], pm[N], position[N+1]
        # And position[N] serves as both eval_after[N-1] and eval_before[N]

        # Collect all board positions at each ply
        ply_boards = [board.copy()]  # position before move 0

        node = game
        move_nodes = []
        while node.variations:
            node = node.variation(0)
            move = node.move

            legal_moves = list(board.legal_moves)

            # Metadata we can derive without evals
            player = headers.get("White", "Unknown") if board.turn == chess.WHITE else headers.get("Black", "Unknown")
            uci_str = move.uci()
            san_str = board.san(move)
            from_sq = chess.square_name(move.from_square)
            to_sq = chess.square_name(move.to_square)
            promo = None if move.promotion is None else chess.Piece(move.promotion, board.turn).symbol()
            piece_obj = board.piece_at(chess.parse_square(from_sq))
            piece_str = piece_obj.symbol() if piece_obj else "?"
            color_str = "White" if board.turn == chess.WHITE else "Black"
            fen_before = board.fen()

            # Collect possible-move boards (from position before push)
            pm_boards = []
            for lm in legal_moves:
                bc = board.copy()
                bc.push(lm)
                pm_boards.append(bc)

            board.push(move)
            moves_list_copy = moves_list + [uci_str]
            moves_list.append(uci_str)

            move_metadata.append({
                'move': move,
                'node': node,
                'player': player,
                'uci': uci_str,
                'san': san_str,
                'from_square': from_sq,
                'to_square': to_sq,
                'promotion': promo,
                'piece': piece_str,
                'color': color_str,
                'fen_before': fen_before,
                'fen_after': board.fen(),
                'move_no': board.ply(),
                'move_no_pair': board.fullmove_number,
                'legal_moves': legal_moves,
                'pm_boards': pm_boards,
                'game_to_position': ' '.join(moves_list_copy),
            })

            ply_boards.append(board.copy())  # position after this move

        # Phase 2: Build flat batch — ply positions + all possible-move positions
        # ply_boards has N+1 entries for N moves
        # eval_before[i] = ply_boards[i], eval_after[i] = ply_boards[i+1]

        if self._use_batch and len(move_metadata) > 0:
            batch_boards = list(ply_boards)  # ply positions first
            pm_offsets = []  # (start_idx, count) in batch_boards for each move's PMs

            for md in move_metadata:
                start = len(batch_boards)
                batch_boards.extend(md['pm_boards'])
                pm_offsets.append((start, len(md['pm_boards'])))

            all_evals = self.evaluator.evaluate_batch(batch_boards)

            ply_evals = all_evals[:len(ply_boards)]
        else:
            # Sequential fallback
            ply_evals = [self.evaluator.evaluate_position(b) for b in ply_boards]
            all_evals = None
            pm_offsets = None

        # Phase 3: Write results
        for i, md in enumerate(move_metadata):
            eval_before_result = ply_evals[i]
            eval_after_result = ply_evals[i + 1]

            (eval_before, mate_count_before) = self._parse_eval(eval_before_result)
            static_eval_before = eval_before_result.get('ev')
            wdl_before = eval_before_result.get('wdl', [333, 334, 333])

            (eval_after, mate_count_after) = self._parse_eval(eval_after_result)
            static_eval_after = eval_after_result.get('ev')
            wdl_after = eval_after_result.get('wdl', [333, 334, 333])

            # Possible moves
            if self._use_batch:
                pm_start, pm_count = pm_offsets[i]
                pm_evals = all_evals[pm_start:pm_start + pm_count]
            else:
                pm_evals = [self.evaluator.evaluate_position(b) for b in md['pm_boards']]

            # Reconstruct board_before for _build_possible_move
            board_before = ply_boards[i]
            for legal_move, pm_eval in zip(md['legal_moves'], pm_evals):
                pm_record = self._build_possible_move(game_id, board_before, legal_move, md['fen_before'], pm_eval)

                if conn and not self.headers_only_sqlite:
                    pm_cols = ', '.join(pm_record.keys())
                    pm_ph = ', '.join(f':{k}' for k in pm_record.keys())
                    conn.execute(f"INSERT INTO possible_move_evals ({pm_cols}) VALUES ({pm_ph})", pm_record)

                if self.write_parquet:
                    self.parquet_writers['possible_moves'].add_possible_move(PossibleMoveRecord(**pm_record))

                possible_moves_count += 1

            time_remaining = self._get_time_remaining(md['node'])
            time_spent = self._get_time_spent(md['node'])

            wdl_total = 1000.0

            move_data = {
                'game_id': game_id,
                'move_no': md['move_no'],
                'move_no_pair': md['move_no_pair'],
                'player': self._hash_player(md['player']),
                'notation': md['san'],
                'move': md['uci'],
                'from_square': md['from_square'],
                'to_square': md['to_square'],
                'piece': md['piece'],
                'promotion': md['promotion'],
                'color': md['color'],
                'fen_before': md['fen_before'],
                'fen_after': md['fen_after'],
                'time_remaining': time_remaining,
                'time_spent': time_spent,
                'game_to_position': md['game_to_position'],
                'white_win_perc_before': wdl_before[0] / wdl_total,
                'black_win_perc_before': wdl_before[2] / wdl_total,
                'draw_perc_before': wdl_before[1] / wdl_total,
                'white_win_perc_after': wdl_after[0] / wdl_total,
                'black_win_perc_after': wdl_after[2] / wdl_total,
                'draw_perc_after': wdl_after[1] / wdl_total,
                'static_eval_before': static_eval_before,
                'static_eval_after': static_eval_after,
                'eval_before': eval_before,
                'mate_count_before': mate_count_before,
                'eval_after': eval_after,
                'mate_count_after': mate_count_after,
                'evaluated_by': 'lc0',
                'evaluator_version': self.engine_info['version'],
            }

            # SQLite gets everything including planes (generated here for SQLite only)
            if conn and not self.headers_only_sqlite:
                move_history = []
                game_moves_so_far = md['game_to_position'].split()
                for past_uci in reversed(game_moves_so_far[:-1]):
                    past_move = chess.Move.from_uci(past_uci)
                    move_history.append((past_move.from_square, past_move.to_square))
                    if len(move_history) >= 2:
                        break
                planes_blob = pack_planes(board_to_planes(ply_boards[i], move_history))
                history_after = [(md['move'].from_square, md['move'].to_square)] + move_history[:1]
                planes_after_blob = pack_planes(board_to_planes(ply_boards[i+1], history_after))
                sqlite_move = dict(move_data)
                sqlite_move['planes_before'] = planes_blob
                sqlite_move['planes_after'] = planes_after_blob
                cols = ', '.join(sqlite_move.keys())
                ph = ', '.join(f':{k}' for k in sqlite_move.keys())
                conn.execute(f"INSERT INTO actual_moves ({cols}) VALUES ({ph})", sqlite_move)

            if self.write_parquet:
                self.parquet_writers['moves'].add_move(MoveRecord(**move_data))

            positions += 1

        return {
            'positions': positions,
            'possible_moves': possible_moves_count,
        }

    # ── process PGN file ────────────────────────────────────────────────

    def process_pgn_file(self, pgn_path: str, max_games: Optional[int] = None) -> Dict:
        pgn_path = Path(pgn_path)
        if not pgn_path.exists():
            raise FileNotFoundError(f"PGN file not found: {pgn_path}")

        print(f"\n{'='*70}")
        print(f"Processing {pgn_path.name} with LC0 ({self.engine_info['version']})")
        print(f"{'='*70}")

        conn = None
        if self.write_sqlite:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")
            conn.execute("PRAGMA busy_timeout=30000")

        games_processed = 0
        games_skipped = 0
        positions_evaluated = 0
        possible_moves_written = 0
        game_order = 0

        start_time = datetime.now()

        try:
            with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn:
                while True:
                    if max_games and games_processed >= max_games:
                        break

                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break

                    game_order += 1
                    game_id = self._game_hash(game)

                    if self.deduplicator.is_game_processed(game_id):
                        games_skipped += 1
                        if games_skipped % 100 == 0 and self.verbose:
                            print(f"  Skipped {games_skipped} duplicates...")
                        continue

                    try:
                        result = self._process_game(
                            conn=conn,
                            game=game,
                            game_id=game_id,
                            game_order=game_order,
                        )

                        positions_evaluated += result['positions']
                        possible_moves_written += result['possible_moves']
                        games_processed += 1

                        self.deduplicator.mark_game_processed(
                            game_id=game_id,
                            file_path=str(pgn_path),
                            metadata={
                                'engine': 'lc0',
                                'version': self.engine_info['version'],
                                'positions': result['positions'],
                                'possible_moves': result['possible_moves'],
                            }
                        )

                        if games_processed % 10 == 0 and self.verbose:
                            elapsed = (datetime.now() - start_time).total_seconds()
                            rate = positions_evaluated / elapsed if elapsed > 0 else 0
                            games_per_sec = games_processed / elapsed if elapsed > 0 else 0
                            eta_sec = (250000 - games_processed) / games_per_sec if games_per_sec > 0 else 0
                            eta_h = eta_sec / 3600
                            print(f"  [{games_processed:,} games | {games_skipped:,} skipped] "
                                  f"{positions_evaluated:,} pos, "
                                  f"{possible_moves_written:,} pm | "
                                  f"{rate:.0f} pos/s, {games_per_sec:.2f} games/s | "
                                  f"ETA: {eta_h:.1f}h")

                        if conn and games_processed % 50 == 0:
                            conn.commit()

                    except Exception as e:
                        print(f"Error processing game {game_id}: {e}")
                        if conn:
                            conn.rollback()
                        import traceback
                        traceback.print_exc()
                        continue

            if conn:
                conn.commit()

            for writer in self.parquet_writers.values():
                writer.close()

        finally:
            if conn:
                conn.close()

        elapsed = (datetime.now() - start_time).total_seconds()
        stats = self.evaluator.get_stats()

        results = {
            'games_processed': games_processed,
            'games_skipped': games_skipped,
            'positions_evaluated': positions_evaluated,
            'possible_moves_written': possible_moves_written,
            'elapsed_seconds': elapsed,
            'positions_per_second': positions_evaluated / elapsed if elapsed > 0 else 0,
            'engine': self.engine_info,
            'output_locations': {
                'sqlite': self.db_path if self.write_sqlite else None,
                'parquet': str(self.output_dir / 'lc0' / self.engine_info['version']) if self.write_parquet else None,
            },
            'engine_stats': stats,
        }

        print(f"\n{'='*70}")
        print(f"COMPLETED: {pgn_path.name}")
        print(f"  Games: {games_processed:,}  Skipped: {games_skipped:,}")
        print(f"  Positions: {positions_evaluated:,}  Possible moves: {possible_moves_written:,}")
        print(f"  Time: {elapsed/60:.1f}min  Rate: {results['positions_per_second']:.1f} pos/sec")
        print(f"{'='*70}\n")

        return results

    def close(self):
        for writer in self.parquet_writers.values():
            writer.close()
        self.evaluator.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LC0 processor with SQLite + Parquet output')
    parser.add_argument('pgn_file', help='PGN file to process')
    parser.add_argument('--db', default='chess.db', help='SQLite database')
    parser.add_argument('--output-dir', default='output', help='Parquet output directory')
    parser.add_argument('--weights', required=True, help='LC0 weights file')
    parser.add_argument('--engine-path', default=None, help='Path to lc0.exe')
    parser.add_argument('--backend', default='cuda-fp16', help='LC0 backend')
    parser.add_argument('--no-parquet', action='store_true', help='Disable Parquet output')
    parser.add_argument('--no-sqlite', action='store_true', help='Disable SQLite output')
    parser.add_argument('--full-sqlite', action='store_true', help='Write moves/possible_moves to SQLite too (default: headers only)')
    parser.add_argument('--num-engines', type=int, default=16, help='Number of async LC0 engines for batch eval (default: 16)')
    parser.add_argument('--max-games', type=int, help='Max games to process')

    args = parser.parse_args()

    processor = LC0GameProcessorWithParquet(
        db_path=args.db,
        weights_path=args.weights,
        output_dir=args.output_dir,
        engine_path=args.engine_path,
        backend=args.backend,
        write_parquet=not args.no_parquet,
        write_sqlite=not args.no_sqlite,
        headers_only_sqlite=not args.full_sqlite,
        num_engines=args.num_engines,
        verbose=True
    )

    try:
        results = processor.process_pgn_file(args.pgn_file, max_games=args.max_games)
    finally:
        processor.close()
