"""
StockfishEvaluator — depth-14 Stockfish evaluator for chess positions.

Wraps chess.engine.SimpleEngine for Stockfish UCI, evaluates at depth 14,
and writes results to parquet using the same schema as LC0.
"""

import chess
import chess.engine
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class StockfishEvalResult:
    """Result from Stockfish evaluation of a single position."""
    score_cp: Optional[int] = None
    score_mate: Optional[int] = None
    best_move: Optional[str] = None
    best_move_san: Optional[str] = None
    pv: Optional[List[str]] = None
    multipv: Optional[List[dict]] = None
    depth: int = 0
    nodes: int = 0


class StockfishEvaluator:
    """
    Stockfish UCI evaluator at a fixed depth.

    Usage:
        evaluator = StockfishEvaluator("stockfish.exe", depth=14)
        evaluator.start()
        result = evaluator.evaluate_position(board, multipv=3)
        evaluator.quit()
    """

    def __init__(
        self,
        stockfish_path: str,
        depth: int = 14,
        threads: int = 1,
        hash_mb: int = 256,
    ):
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self.version: str = "unknown"

    def start(self):
        """Start the Stockfish engine process."""
        self._engine = chess.engine.SimpleEngine.popen_uci(
            self.stockfish_path,
            timeout=30,
        )
        self._engine.configure({
            "Threads": self.threads,
            "Hash": self.hash_mb,
        })
        # Try to get version from engine id
        try:
            ident = self._engine.id.get("name", "Stockfish unknown")
            self.version = ident
        except Exception:
            self.version = "Stockfish"

    def evaluate_position(
        self, board: chess.Board, multipv: int = 1
    ) -> StockfishEvalResult:
        """Evaluate a position at the configured depth."""
        if self._engine is None:
            raise RuntimeError("Engine not started — call start() first")

        info = self._engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth),
            multipv=multipv,
        )

        result = StockfishEvalResult()

        entries = info if isinstance(info, list) else [info]
        top = entries[0]

        # Parse top line
        score = top.get("score")
        if score:
            pov = score.white()
            if pov.is_mate():
                result.score_mate = pov.mate()
            else:
                result.score_cp = pov.score()

        pv_moves = top.get("pv", [])
        if pv_moves:
            result.best_move = pv_moves[0].uci()
            try:
                result.best_move_san = board.san(pv_moves[0])
            except Exception:
                result.best_move_san = result.best_move
        result.pv = [m.uci() for m in pv_moves]
        result.depth = top.get("depth", self.depth)
        result.nodes = top.get("nodes", 0)

        # Collect all PV lines
        result.multipv = []
        for entry in entries:
            mv = {}
            s = entry.get("score")
            if s:
                p = s.white()
                if p.is_mate():
                    mv["score_mate"] = p.mate()
                    mv["score_cp"] = None
                else:
                    mv["score_cp"] = p.score()
                    mv["score_mate"] = None
            pv = entry.get("pv", [])
            if pv:
                mv["move_uci"] = pv[0].uci()
                try:
                    mv["move_san"] = board.san(pv[0])
                except Exception:
                    mv["move_san"] = mv["move_uci"]
            mv["nodes"] = entry.get("nodes", 0)
            mv["depth"] = entry.get("depth", 0)
            result.multipv.append(mv)

        return result

    def quit(self):
        if self._engine:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.quit()


if __name__ == "__main__":
    import sys

    sf_path = sys.argv[1] if len(sys.argv) > 1 else "stockfish"
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")

    with StockfishEvaluator(sf_path, depth=14) as sf:
        print(f"Engine: {sf.version}")
        result = sf.evaluate_position(board, multipv=3)
        print(f"Best move: {result.best_move_san}  Score: {result.score_cp}cp")
        for i, pv in enumerate(result.multipv or []):
            print(f"  PV {i+1}: {pv.get('move_san')} = {pv.get('score_cp')}cp")
