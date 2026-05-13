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
    from parquet_writer import ParquetWriter
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("Warning: parquet_writer not available. Install pyarrow: pip install pyarrow")


class LC0GameProcessorWithParquet:
    """
    LC0 processor that writes to BOTH SQLite and Parquet.
    Parquet schemas match the old chess_evaluator SQLite tables 1:1.
    """

    def __init__(self,
                 db_path: str,
                 weights_path: str,
                 output_dir: str = "output",
                 backend: str = "cuda-fp16",
                 write_parquet: bool = True,
                 write_sqlite: bool = True,
                 verbose: bool = True):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.write_parquet = write_parquet and PARQUET_AVAILABLE
        self.write_sqlite = write_sqlite
        self.verbose = verbose

        if write_parquet and not PARQUET_AVAILABLE:
            raise ImportError("Parquet writing requested but parquet_writer not available.")

        print(f"Initializing LC0 processor with {backend} backend...")
        self.evaluator = LC0Evaluator(
            weights_path=weights_path,
            backend=backend,
            threads=4,
            max_batch_size=256,
            verbose=False
        )

        self.deduplicator = GameDeduplicator(db_path)

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

        # Games go to parquet only; headers stay in SQLite via chess_evaluator
        if self.write_parquet:
            self.parquet_writers['games'].write(game_data)

        # ── iterate moves ──
        board = game.board()
        positions = 0
        possible_moves_count = 0
        moves_list = []
        prev_boards = []

        node = game
        while node.variations:
            node = node.variation(0)
            move = node.move

            # ── position BEFORE the move ──
            fen_before = board.fen()
            eval_before_result = self.evaluator.evaluate_position(board)
            (eval_before, mate_count_before) = self._parse_eval(eval_before_result)
            static_eval_before = eval_before_result.get('ev')  # LC0 has no separate static eval; use NN eval
            wdl_before = eval_before_result.get('wdl', [333, 334, 333])

            # ── possible moves at this position ──
            legal_moves = list(board.legal_moves)
            for legal_move in legal_moves:
                board_copy = board.copy()
                board_copy.push(legal_move)
                pm_eval = self.evaluator.evaluate_position(board_copy)
                pm_record = self._build_possible_move(game_id, board, legal_move, fen_before, pm_eval)

                if conn:
                    pm_cols = ', '.join(pm_record.keys())
                    pm_ph = ', '.join(f':{k}' for k in pm_record.keys())
                    conn.execute(f"INSERT INTO possible_move_evals ({pm_cols}) VALUES ({pm_ph})", pm_record)

                if self.write_parquet:
                    self.parquet_writers['possible_moves'].write(pm_record)

                possible_moves_count += 1

            # ── derive move metadata BEFORE push ──
            player = headers.get("White", "Unknown") if board.turn == chess.WHITE else headers.get("Black", "Unknown")
            uci = move.uci()
            san = board.san(move)
            from_square = chess.square_name(move.from_square)
            to_square = chess.square_name(move.to_square)
            promotion = None if move.promotion is None else chess.Piece(move.promotion, board.turn).symbol()
            piece_obj = board.piece_at(chess.parse_square(from_square))
            piece = piece_obj.symbol() if piece_obj else "?"
            color = "White" if board.turn == chess.WHITE else "Black"

            # ── planes before ──
            planes = board_to_planes(board, prev_boards)
            planes_blob = pack_planes(planes)

            # ── execute move ──
            board.push(move)
            moves_list.append(uci)

            move_no_pair = board.fullmove_number
            move_no = board.ply()
            fen_after = board.fen()

            # ── position AFTER the move ──
            eval_after_result = self.evaluator.evaluate_position(board)
            (eval_after, mate_count_after) = self._parse_eval(eval_after_result)
            static_eval_after = eval_after_result.get('ev')
            wdl_after = eval_after_result.get('wdl', [333, 334, 333])

            planes_after = board_to_planes(board, [board])
            planes_after_blob = pack_planes(planes_after)

            time_remaining = self._get_time_remaining(node)
            time_spent = self._get_time_spent(node)

            wdl_total = 1000.0

            move_data = {
                'game_id': game_id,
                'move_no': move_no,
                'move_no_pair': move_no_pair,
                'player': self._hash_player(player),
                'notation': san,
                'move': uci,
                'from_square': from_square,
                'to_square': to_square,
                'piece': piece,
                'promotion': promotion,
                'color': color,
                'fen_before': fen_before,
                'fen_after': fen_after,
                'time_remaining': time_remaining,
                'time_spent': time_spent,
                'game_to_position': ' '.join(moves_list),
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

            # SQLite gets everything including planes
            if conn:
                sqlite_move = dict(move_data)
                sqlite_move['planes_before'] = planes_blob
                sqlite_move['planes_after'] = planes_after_blob
                cols = ', '.join(sqlite_move.keys())
                ph = ', '.join(f':{k}' for k in sqlite_move.keys())
                conn.execute(f"INSERT INTO actual_moves ({cols}) VALUES ({ph})", sqlite_move)

            # Parquet gets everything except planes (generate at training time from FENs)
            if self.write_parquet:
                self.parquet_writers['moves'].write(move_data)

            positions += 1
            prev_boards = [board.copy()] + prev_boards[:1]

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
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")

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
                            print(f"  [{games_processed}] {positions_evaluated:,} positions, "
                                  f"{possible_moves_written:,} possible moves, "
                                  f"{rate:.1f} pos/sec")

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
    parser.add_argument('--backend', default='cuda-fp16', help='LC0 backend')
    parser.add_argument('--no-parquet', action='store_true', help='Disable Parquet output')
    parser.add_argument('--no-sqlite', action='store_true', help='Disable SQLite output')
    parser.add_argument('--max-games', type=int, help='Max games to process')

    args = parser.parse_args()

    processor = LC0GameProcessorWithParquet(
        db_path=args.db,
        weights_path=args.weights,
        output_dir=args.output_dir,
        backend=args.backend,
        write_parquet=not args.no_parquet,
        write_sqlite=not args.no_sqlite,
        verbose=True
    )

    try:
        results = processor.process_pgn_file(args.pgn_file, max_games=args.max_games)
    finally:
        processor.close()
