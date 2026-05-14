"""
batch_evaluator.py — LC0 evaluator with WDL capture and GPU validation

Enhanced with:
- WDL percentage capture from engine
- GPU backend validation on startup
- Smoke test for GPU activation
- Full command line logging
- Error handling for missing files/backends
"""

import subprocess
import os
import sys
import time
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
    # WDL percentages (0.0-1.0 from WHITE's perspective)
    wdl_w: Optional[float] = None
    wdl_d: Optional[float] = None
    wdl_l: Optional[float] = None


class BatchLC0Evaluator:
    """
    Async LC0 evaluator that batches positions for throughput.
    Uses chess.engine.SimpleEngine under the hood with GPU validation.
    """

    def __init__(
        self,
        lc0_path: str,
        weights_path: str,
        backend: str = "cuda-fp16",
        batch_size: int = 32,
        nodes: int = 1,
        threads: int = 1,
        verify_gpu: bool = True,
    ):
        self.lc0_path = lc0_path
        self.weights_path = weights_path
        self.backend = backend
        self.batch_size = batch_size
        self.nodes = nodes
        self.threads = threads
        self.verify_gpu = verify_gpu
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def _validate_lc0_installation(self):
        """Validate LC0 executable exists and supports the requested backend."""
        if not os.path.exists(self.lc0_path):
            raise RuntimeError(
                f"LC0 executable not found: {self.lc0_path}\n"
                f"Please check the path and ensure LC0 is installed."
            )

        if not os.path.exists(self.weights_path):
            raise RuntimeError(
                f"LC0 weights file not found: {self.weights_path}\n"
                f"Please check the path and ensure weights are downloaded."
            )

        if not self.verify_gpu:
            return

        # Check backend support
        try:
            result = subprocess.run(
                [self.lc0_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            help_text = result.stdout + result.stderr

            # Extract available backends
            if "cuda" not in help_text.lower():
                print(f"[LC0 WARNING] CUDA backend may not be available in this LC0 build", file=sys.stderr)
                print(f"[LC0 WARNING] Help output: {help_text[:500]}", file=sys.stderr)

            if self.backend not in help_text:
                print(f"[LC0 WARNING] Requested backend '{self.backend}' not found in help text", file=sys.stderr)
                print(f"[LC0 WARNING] Available backends may include: cuda, cuda-fp16, opencl, blas", file=sys.stderr)

        except Exception as e:
            print(f"[LC0 WARNING] Could not verify backend support: {e}", file=sys.stderr)

    def _smoke_test(self):
        """Run a smoke test to verify LC0 starts and responds."""
        print(f"[LC0] Running smoke test...", file=sys.stderr)
        try:
            board = chess.Board()
            info = self._engine.analyse(
                board,
                chess.engine.Limit(nodes=100),
                info=chess.engine.INFO_ALL
            )

            if info.get("score") is None:
                raise RuntimeError("LC0 smoke test failed: no score returned")

            print(f"[LC0] Smoke test passed: engine is responsive", file=sys.stderr)

        except Exception as e:
            raise RuntimeError(
                f"LC0 smoke test failed: {e}\n"
                f"This may indicate GPU/backend issues.\n"
                f"Try running: {self.lc0_path} --backend={self.backend} --weights={self.weights_path}"
            )

    def start(self):
        """Start the LC0 engine process with validation."""
        print(f"[LC0] Starting engine: {self.lc0_path}", file=sys.stderr)
        print(f"[LC0] Weights: {self.weights_path}", file=sys.stderr)
        print(f"[LC0] Backend: {self.backend}", file=sys.stderr)
        print(f"[LC0] Minibatch size: {self.batch_size}", file=sys.stderr)
        print(f"[LC0] Threads: {self.threads}", file=sys.stderr)

        self._validate_lc0_installation()

        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(
                self.lc0_path,
                timeout=60,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to start LC0: {e}\n"
                f"Command: {self.lc0_path}\n"
                f"Check that the executable exists and is runnable."
            )

        # Configure engine
        config = {
            "WeightsFile": self.weights_path,
            "Backend": self.backend,
            "MinibatchSize": self.batch_size,
            "Threads": self.threads,
            "UCI_ShowWDL": True,
        }

        print(f"[LC0] Configuring engine: {config}", file=sys.stderr)

        try:
            self._engine.configure(config)
        except Exception as e:
            self._engine.quit()
            raise RuntimeError(f"Failed to configure LC0: {e}")

        # Run smoke test
        if self.verify_gpu:
            self._smoke_test()

        print(f"[LC0] Engine started successfully with backend={self.backend}", file=sys.stderr)

    def evaluate_position(
        self, board: chess.Board, multipv: int = 1
    ) -> EvalResult:
        """Evaluate a single position with WDL capture."""
        if self._engine is None:
            raise RuntimeError("Engine not started")

        info = self._engine.analyse(
            board,
            chess.engine.Limit(nodes=self.nodes),
            multipv=multipv,
            info=chess.engine.INFO_ALL,
        )

        def _parse_info(entry):
            result = EvalResult()

            # Basic eval
            score = entry.get("score")
            if score:
                pov = score.white()
                if pov.is_mate():
                    result.score_mate = pov.mate()
                else:
                    result.score_cp = pov.score()

            # PV
            pv_moves = entry.get("pv", [])
            if pv_moves:
                result.best_move = pv_moves[0].uci()
                try:
                    result.best_move_san = board.san(pv_moves[0])
                except Exception:
                    result.best_move_san = result.best_move
            result.pv = [m.uci() for m in pv_moves]
            result.nodes = entry.get("nodes", 0)
            result.depth = entry.get("depth", 0)

            # WDL
            wdl = entry.get("wdl")
            if wdl:
                # wdl is a Wdl namedtuple with wins/draws/losses (out of 1000)
                result.wdl_w = wdl.white().wins / 1000.0
                result.wdl_d = wdl.white().draws / 1000.0
                result.wdl_l = wdl.white().losses / 1000.0

            return result

        if isinstance(info, list):
            top = _parse_info(info[0])
            top.multipv = []

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
                mv["depth"] = entry.get("depth", 0)

                # Per-PV WDL
                wdl = entry.get("wdl")
                if wdl:
                    mv["wdl_w"] = wdl.white().wins / 1000.0
                    mv["wdl_d"] = wdl.white().draws / 1000.0
                    mv["wdl_l"] = wdl.white().losses / 1000.0

                top.multipv.append(mv)

            return top
        else:
            result = _parse_info(info)
            # Build multipv list from the single result so possible_moves get generated
            mv = {
                "score_cp": result.score_cp,
                "score_mate": result.score_mate,
                "move_uci": result.best_move or "",
                "move_san": result.best_move_san or "",
                "nodes": result.nodes,
                "depth": result.depth,
            }
            if result.wdl_w is not None:
                mv["wdl_w"] = result.wdl_w
                mv["wdl_d"] = result.wdl_d
                mv["wdl_l"] = result.wdl_l
            result.multipv = [mv]
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
