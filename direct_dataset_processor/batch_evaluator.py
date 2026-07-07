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
  - CPuct=100.0: maximum exploration bias to ensure all legal moves get visited
  - PolicyTemperature=10.0: flatten policy for uniform coverage

Node budget per position:
  node_limit = clamp(n_legal * nodes_mult, min_nodes, max_nodes)
  Default: nodes_mult=1.0, min_nodes=1, max_nodes=0 (no cap)
  → nodes = n_legal (one visit per legal move per PV counter)

Speed: One engine.analyse() call per position (~5-15ms on RTX 4090)
instead of ~30 separate calls (~150ms+). GPU batches all NN evals
internally.
"""

import chess
import chess.engine
from typing import List, Optional, Dict
from dataclasses import dataclass


# Maximum possible legal moves in chess (theoretical)
MAX_LEGAL_MOVES = 218


def _extract_wdl(info_entry: dict):
    """Extract WDL from an engine info dict entry, returning (w, d, l) as 0.0-1.0 or (None, None, None).

    Always returns White's perspective by calling .white() on the PovWdl object.
    """
    wdl_pov = info_entry.get("wdl")
    if wdl_pov is not None:
        wdl = wdl_pov.white()
        return wdl.wins / 1000.0, wdl.draws / 1000.0, wdl.losses / 1000.0
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

    Node budget: node_limit = clamp(n_legal * nodes_mult, min_nodes, max_nodes)
    Default produces nodes = n_legal (dynamic PV = number of legal moves).
    """

    def __init__(
        self,
        lc0_path: str,
        weights_path: str,
        backend: str = "cuda-fp16",
        batch_size: int = 256,
        threads: int = 1,
        min_nodes: int = 1,
        max_nodes: int = 0,
        nodes_mult: float = 1.0,
    ):
        self.lc0_path = lc0_path
        self.weights_path = weights_path
        self.backend = backend
        self.batch_size = batch_size
        self.threads = threads
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.nodes_mult = nodes_mult
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def _compute_node_limit(self, n_legal: int) -> int:
        """Compute node budget: clamp(n_legal * nodes_mult, min_nodes, max_nodes)."""
        node_limit = max(self.min_nodes, int(n_legal * self.nodes_mult))
        if self.max_nodes > 0:
            node_limit = min(node_limit, self.max_nodes)
        return node_limit

    def start(self):
        """Start the LC0 engine process with all-legal-moves exploration settings."""
        self._engine = chess.engine.SimpleEngine.popen_uci(
            self.lc0_path,
            timeout=60,
        )
        self._engine.configure({
            "WeightsFile": self.weights_path,
            "Backend": self.backend,
            "MinibatchSize": self.batch_size,
            "Threads": self.threads,
            "UCI_ShowWDL": True,
            "PerPVCounters": True,
            "SmartPruningFactor": 0,
            "FpuStrategy": "absolute",
            "FpuValue": 0,
            "CPuct": 100.0,
            "PolicyTemperature": 10.0,
        })

    def evaluate_position(
        self, board: chess.Board, multipv: int = 1
    ) -> EvalResult:
        """Evaluate a single position. Returns scores from WHITE's perspective.

        Uses the same dynamic node budget as evaluate_all_legal_moves.
        """
        if self._engine is None:
            raise RuntimeError("Engine not started")

        n_legal = len(list(board.legal_moves))
        node_limit = self._compute_node_limit(max(n_legal, 1))

        info = self._engine.analyse(
            board,
            chess.engine.Limit(nodes=node_limit),
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

        Node budget: clamp(n_legal * nodes_mult, min_nodes, max_nodes).
        Default: nodes = n_legal (one visit per PV counter).

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
        """
        if self._engine is None:
            raise RuntimeError("Engine not started")

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        n_legal = len(legal_moves)

        # Pre-compute SAN and fen_after for all legal moves (instant, no engine)
        move_info = {}
        for move in legal_moves:
            try:
                san = board.san(move)
            except Exception:
                san = move.uci()
            board_copy = board.copy()
            board_copy.push(move)
            move_info[move.uci()] = {
                "san": san,
                "fen_after": board_copy.fen(),
            }

        node_limit = self._compute_node_limit(n_legal)
        infos = self._engine.analyse(
            board,
            chess.engine.Limit(nodes=node_limit),
            multipv=MAX_LEGAL_MOVES,
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

        # Safety fallback: if any legal moves were missed (shouldn't happen
        # with PerPVCounters + exploration settings), eval them individually
        if len(results) < n_legal:
            fallback_limit = self._compute_node_limit(1)
            for move in legal_moves:
                uci = move.uci()
                if uci in seen_moves:
                    continue
                mi = move_info[uci]
                board_after = chess.Board(mi["fen_after"])
                entry = {
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
                }
                if board_after.is_game_over():
                    if board_after.is_checkmate():
                        if board_after.turn == chess.WHITE:
                            entry.update(score_cp=-10000, score_mate=0,
                                         wdl_w=0.0, wdl_d=0.0, wdl_l=1.0)
                        else:
                            entry.update(score_cp=10000, score_mate=0,
                                         wdl_w=1.0, wdl_d=0.0, wdl_l=0.0)
                    else:
                        entry.update(score_cp=0, wdl_w=0.0, wdl_d=1.0, wdl_l=0.0)
                else:
                    try:
                        fallback = self._engine.analyse(
                            board_after,
                            chess.engine.Limit(nodes=fallback_limit),
                            multipv=1,
                            info=chess.engine.INFO_ALL,
                        )
                        top = fallback[0] if isinstance(fallback, list) else fallback
                        s = top.get("score")
                        if s:
                            p = s.white()
                            if p.is_mate():
                                entry["score_mate"] = p.mate()
                            else:
                                entry["score_cp"] = p.score()
                        fw, fd, fl = _extract_wdl(top)
                        entry["wdl_w"] = fw
                        entry["wdl_d"] = fd
                        entry["wdl_l"] = fl
                    except Exception:
                        pass
                results.append(entry)

        return results

    def quit(self):
        if self._engine:
            try:
                self._engine.quit()
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

    def is_alive(self) -> bool:
        return self._evaluator._engine is not None

    def restart(self):
        self._evaluator.quit()
        self._evaluator.start()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.quit()
