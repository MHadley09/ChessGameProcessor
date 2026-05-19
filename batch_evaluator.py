"""
BatchLC0Evaluator and SyncBatchEvaluator — LC0 engine wrapper with WDL support.

Evaluates positions via LC0 using chess.engine.SimpleEngine.
Uses a SINGLE analyse() call with multipv=218 and PerPVCounters=True to
evaluate ALL legal moves in one shot. Each PV gets its own independent
search tree, giving genuine per-move eval + WDL.

Key LC0 settings for all-legal-moves coverage:
  - PerPVCounters=True: each PV builds its own search tree
  - SmartPruningFactor=0: no pruning of unpromising moves
  - FpuStrategy=absolute, FpuValue=0: neutral first-play urgency
  - CPuct=5.0: heavy exploration bias
  - PolicyTemperature=10.0: flatten policy for uniform coverage

Speed: One engine.analyse() call per position with nodes=100 and
multipv=218 (~5-15ms on RTX 4090). GPU batches all NN evals internally.
~10-30x faster than individual per-move UCI calls.
"""

import chess
import chess.engine

_VERSION = "uci-optimized-v5-nn-cache-flag"
from typing import List, Optional, Dict
from dataclasses import dataclass


# Maximum possible legal moves in chess (theoretical)
MAX_LEGAL_MOVES = 218


def _extract_wdl(info_entry: dict):
    """Extract WDL from an engine info dict entry, returning (w, d, l) as 0.0-1.0 or (None, None, None).
    python-chess returns PovWdl — must call .white() to get raw Wdl with .wins/.draws/.losses."""
    pov_wdl = info_entry.get("wdl")
    if pov_wdl is not None:
        w = pov_wdl.white()
        return w.wins / 1000.0, w.draws / 1000.0, w.losses / 1000.0
    return None, None, None


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
    # WDL from WHITE's perspective, 0.0-1.0
    wdl_w: Optional[float] = None
    wdl_d: Optional[float] = None
    wdl_l: Optional[float] = None


class BatchLC0Evaluator:
    """
    LC0 evaluator using single-call multipv for all-legal-moves evaluation.
    """

    def __init__(
        self,
        lc0_path: str,
        weights_path: str,
        backend: str = "cuda-fp16",
        batch_size: int = 256,
        nodes: int = 1,
        multipv_nodes: int = 100,  # Legacy default; now using dynamic nodes (3x legal moves, cap 150, else 2x cap 250)
        nn_cache_size: int = 50000,
        threads: int = 1,
    ):
        self.lc0_path = lc0_path
        self.weights_path = weights_path
        self.backend = backend
        self.batch_size = batch_size
        self.nodes = nodes                  # nodes for single-position eval (evaluate_position)
        self.multipv_nodes = multipv_nodes  # nodes for all-legal-moves eval (evaluate_all_legal_moves)
        self.nn_cache_size = nn_cache_size
        self.threads = threads
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def start(self):
        """Start the LC0 engine process with all-legal-moves exploration settings."""
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(
                self.lc0_path,
                timeout=60,
            )
        except Exception as e:
            self._engine = None
            raise RuntimeError(f"Failed to start LC0 engine: {e}") from e
        self._engine.configure({
            "WeightsFile": self.weights_path,
            "Backend": self.backend,
            "MinibatchSize": self.batch_size,
            "Threads": self.threads,
            # All-legal-moves settings: force MCTS to visit every child
            "UCI_ShowWDL": True,
            "PerPVCounters": True,
            "SmartPruningFactor": 0,
            "FpuStrategy": "absolute",
            "FpuValue": 0,
            "CPuct": 5.0,
            "PolicyTemperature": 10.0,
            # Performance tuning
            "NNCacheSize": self.nn_cache_size,  # Configurable via flag
            "OutOfOrderEval": True,   # Better GPU utilization
            "MaxCollisionEvents": 32, # Reduce search thread contention
        })

    def evaluate_position(
        self, board: chess.Board, multipv: int = 1
    ) -> EvalResult:
        """Evaluate a single position. Returns scores from WHITE's perspective."""
        if self._engine is None:
            raise RuntimeError("Engine not started")

        info = self._engine.analyse(
            board,
            chess.engine.Limit(nodes=self.nodes),
            multipv=multipv,
            info=chess.engine.INFO_ALL,
        )

        entries = info if isinstance(info, list) else [info]
        top = entries[0]

        result = EvalResult()

        score = top.get("score")
        if score:
            pov = score.white()
            if pov.is_mate():
                result.score_mate = pov.mate()
            else:
                result.score_cp = pov.score()

        w, d, l = _extract_wdl(top)
        result.wdl_w = w
        result.wdl_d = d
        result.wdl_l = l

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

            pw, pd, pl = _extract_wdl(entry)
            mv["wdl_w"] = pw
            mv["wdl_d"] = pd
            mv["wdl_l"] = pl

            result.multipv.append(mv)

        return result

    def evaluate_all_legal_moves(self, board: chess.Board) -> List[Dict]:
        """
        Evaluate EVERY legal move in ONE engine call.

        Uses multipv=218 with PerPVCounters=True. Each PV gets its own
        independent search tree, so every legal move receives a genuine
        NN evaluation with unique WDL — all from a single analyse() call.

        Returns a list of dicts, one per legal move:
            {
                "move_uci": str,       # e.g. "e2e4"
                "move_san": str,       # e.g. "e4"
                "score_cp": int|None,  # centipawns from WHITE's perspective
                "score_mate": int|None,
                "nodes": int,
                "depth": int,
                "wdl_w": float|None,   # white win prob 0.0-1.0
                "wdl_d": float|None,   # draw prob
                "wdl_l": float|None,   # white loss prob (= black win)
                "fen_after": str,      # FEN of the position after this move
            }

        Speed: ONE analyse() call per position (~5-15ms on RTX 4090).
        GPU batches all NN evals internally. ~10-30x faster than
        individual per-move calls.

        Uses self.multipv_nodes (default 250) as the node budget.
        With PerPVCounters=True, 250 nodes across ~30 legal moves gives
        ~8 visits per move on average — enough for stable per-move WDL.
        """
        if self._engine is None:
            raise RuntimeError("Engine not started")

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        n_legal = len(legal_moves)

        # Pre-compute SAN and fen_after for all legal moves (instant, no engine)
        # Optimized: use push/pop instead of board.copy()
        move_info = {}
        for move in legal_moves:
            try:
                san = board.san(move)
            except Exception:
                san = move.uci()
            board.push(move)
            fen_after = board.fen()
            board.pop()
            move_info[move.uci()] = {
                "san": san,
                "fen_after": fen_after,
            }

        # Dynamic nodes: 5x legal moves (cap 300), else 3x (cap 500), floor 150
        # Higher node counts for better accuracy
        n_legal = len(legal_moves)
        nodes = 5 * n_legal
        if nodes > 300:
            nodes = min(3 * n_legal, 500)
        nodes = max(nodes, 150)  # Floor at 150 nodes
        
        # Single call: multipv=n_legal (exact match to legal moves)
        # PerPVCounters=True gives each PV its own search tree
        infos = self._engine.analyse(
            board,
            chess.engine.Limit(nodes=nodes),
            multipv=n_legal,
            info=chess.engine.INFO_ALL,
        )

        if not isinstance(infos, list):
            infos = [infos]

        results = []
        seen_moves = set()

        for info in infos:
            pv = info.get("pv")
            if not pv:
                continue

            move = pv[0]
            uci = move.uci()
            if uci in seen_moves:
                continue
            seen_moves.add(uci)

            mi = move_info.get(uci, {"san": uci, "fen_after": ""})

            entry = {
                "move_uci": uci,
                "move_san": mi["san"],
                "score_cp": None,
                "score_mate": None,
                "nodes": info.get("nodes", 0),
                "depth": info.get("depth", 0),
                "wdl_w": None,
                "wdl_d": None,
                "wdl_l": None,
                "fen_after": mi["fen_after"],
            }

            score = info.get("score")
            if score:
                pov = score.white()
                if pov.is_mate():
                    entry["score_mate"] = pov.mate()
                else:
                    entry["score_cp"] = pov.score()

            w, d, l = _extract_wdl(info)
            entry["wdl_w"] = w
            entry["wdl_d"] = d
            entry["wdl_l"] = l

            results.append(entry)

        # Warn if some legal moves were missed (rare edge case)
        if len(results) < n_legal:
            missing = [m.uci() for m in legal_moves if m.uci() not in seen_moves]
            import sys
            print(
                f"[WARN] multipv={n_legal} returned {len(results)}/{n_legal} "
                f"legal moves. Missing: {missing[:5]}",
                file=sys.stderr,
            )
            # Fill missing moves with None evals so downstream schema stays consistent
            for move in legal_moves:
                uci = move.uci()
                if uci not in seen_moves:
                    mi = move_info.get(uci, {"san": uci, "fen_after": ""})
                    results.append({
                        "move_uci": uci,
                        "move_san": mi["san"],
                        "score_cp": None,
                        "score_mate": None,
                        "nodes": 0,
                        "depth": 0,
                        "wdl_w": None,
                        "wdl_d": None,
                        "wdl_l": None,
                        "fen_after": mi["fen_after"],
                    })

        return results

    def quit(self):
        if self._engine:
            try:
                self._engine.quit()
            except Exception:
                try:
                    self._engine.close()
                except Exception:
                    pass
            self._engine = None


class SyncBatchEvaluator:
    """Synchronous wrapper around BatchLC0Evaluator."""

    def __init__(self, *args, **kwargs):
        self._evaluator = BatchLC0Evaluator(*args, **kwargs)

    def start(self):
        self._evaluator.start()

    def evaluate_position(self, board: chess.Board, multipv: int = 1) -> EvalResult:
        return self._evaluator.evaluate_position(board, multipv=multipv)

    def evaluate_all_legal_moves(self, board: chess.Board) -> List[Dict]:
        return self._evaluator.evaluate_all_legal_moves(board)

    def quit(self):
        self._evaluator.quit()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.quit()
