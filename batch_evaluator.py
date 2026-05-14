"""
BatchLC0Evaluator and SyncBatchEvaluator — copied from repo for completeness.

Batches multiple positions and sends them to LC0 in a single `go nodes`
request using the multipv/searchmoves approach.  SyncBatchEvaluator wraps
the async version so callers can use it from synchronous code.
"""

import asyncio
import threading
import subprocess
import chess
import chess.engine
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Result of evaluating a single position."""
    score_cp: Optional[int] = None
    score_mate: Optional[int] = None
    best_move: Optional[str] = None
    best_move_san: Optional[str] = None
    pv: Optional[List[str]] = None
    multipv: Optional[List[dict]] = None
    nodes: int = 0
    depth: int = 0


class BatchLC0Evaluator:
    """
    Async LC0 evaluator that batches positions for throughput.
    Uses chess.engine.SimpleEngine under the hood.
    """

    def __init__(
        self,
        lc0_path: str,
        weights_path: str,
        backend: str = "cuda-fp16",
        batch_size: int = 32,
        nodes: int = 1,
        threads: int = 1,
    ):
        self.lc0_path = lc0_path
        self.weights_path = weights_path
        self.backend = backend
        self.batch_size = batch_size
        self.nodes = nodes
        self.threads = threads
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def start(self):
        """Start the LC0 engine process."""
        self._engine = chess.engine.SimpleEngine.popen_uci(
            self.lc0_path,
            timeout=60,
        )
        self._engine.configure({
            "WeightsFile": self.weights_path,
            "Backend": self.backend,
            "MinibatchSize": self.batch_size,
            "Threads": self.threads,
        })

    def evaluate_position(
        self, board: chess.Board, multipv: int = 1
    ) -> EvalResult:
        """Evaluate a single position."""
        if self._engine is None:
            raise RuntimeError("Engine not started")

        info = self._engine.analyse(
            board,
            chess.engine.Limit(nodes=self.nodes),
            multipv=multipv,
        )

        if isinstance(info, list):
            top = info[0]
            result = EvalResult()
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
            result.nodes = top.get("nodes", 0)
            result.depth = top.get("depth", 0)

            # Collect all PVs
            result.multipv = []
            for entry in info:
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
                result.multipv.append(mv)

            return result
        else:
            result = EvalResult()
            score = info.get("score")
            if score:
                pov = score.white()
                if pov.is_mate():
                    result.score_mate = pov.mate()
                else:
                    result.score_cp = pov.score()
            pv_moves = info.get("pv", [])
            if pv_moves:
                result.best_move = pv_moves[0].uci()
                try:
                    result.best_move_san = board.san(pv_moves[0])
                except Exception:
                    result.best_move_san = result.best_move
            result.pv = [m.uci() for m in pv_moves]
            result.nodes = info.get("nodes", 0)
            result.depth = info.get("depth", 0)
            return result

    def quit(self):
        if self._engine:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None


class SyncBatchEvaluator:
    """
    Synchronous wrapper around BatchLC0Evaluator.
    Runs the engine in its own thread with a private event loop so it
    works on Windows where the default event loop is not re-entrant.
    """

    def __init__(self, *args, **kwargs):
        self._evaluator = BatchLC0Evaluator(*args, **kwargs)

    def start(self):
        self._evaluator.start()

    def evaluate_position(self, board: chess.Board, multipv: int = 1) -> EvalResult:
        return self._evaluator.evaluate_position(board, multipv=multipv)

    def quit(self):
        self._evaluator.quit()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.quit()
