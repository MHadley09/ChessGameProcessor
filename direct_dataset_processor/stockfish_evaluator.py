"""
DirectStockfishEvaluator — Direct UCI subprocess wrapper for Stockfish.

Same interface and return format as DirectLC0Evaluator / SyncDirectEvaluator.
Uses MultiPV to evaluate all legal moves in a single search call.
Supports depth, nodes, and time-based search limits (configurable).

No GPU required — runs on CPU, so you can run many more workers in parallel
than with LC0.

Drop-in replacement: same evaluate_all_legal_moves() return format.
"""

import subprocess
import threading
import chess
from typing import List, Dict, Optional


class DirectStockfishEvaluator:
    """
    Stockfish evaluator using direct UCI subprocess communication.

    Evaluates all legal moves via MultiPV. Supports configurable search
    limits: depth, nodes, and/or movetime. If multiple are set, Stockfish
    stops at whichever limit is reached first.

    Returns the same format as DirectLC0Evaluator for pipeline compatibility.
    """

    def __init__(
        self,
        stockfish_path: str,
        threads: int = 1,
        hash_mb: int = 128,
        max_depth: int = 0,
        max_nodes: int = 0,
        movetime_ms: int = 0,
        # Accepted but ignored — keeps interface compatible with LC0 worker args
        lc0_path: str = "",
        weights_path: str = "",
        backend: str = "",
        batch_size: int = 0,
        min_nodes: int = 0,
        nodes_mult: float = 0,
    ):
        self.stockfish_path = stockfish_path
        self.threads = threads
        self.hash_mb = hash_mb
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.movetime_ms = movetime_ms

        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._multipv_set: Optional[int] = None

    def start(self):
        """Start the Stockfish engine process and configure UCI options."""
        self._proc = subprocess.Popen(
            [self.stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

        # UCI handshake
        self._send("uci")
        self._read_until("uciok")

        # Configure
        self._set_option("Threads", str(self.threads))
        self._set_option("Hash", str(self.hash_mb))
        self._set_option("UCI_ShowWDL", "true")

        self._send("isready")
        self._read_until("readyok")

    def _drain_stderr(self):
        """Read and discard stderr to prevent pipe blocking."""
        try:
            while self._proc and self._proc.stderr:
                line = self._proc.stderr.readline()
                if not line:
                    break
        except (ValueError, OSError):
            pass

    def _send(self, cmd: str):
        """Send a UCI command."""
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _set_option(self, name: str, value: str):
        """Send a setoption command."""
        self._send(f"setoption name {name} value {value}")

    def _read_until(self, stop_token: str) -> List[str]:
        """Read stdout lines until a line starts with stop_token."""
        lines = []
        while True:
            line = self._proc.stdout.readline().strip()
            if not line and self._proc.poll() is not None:
                raise RuntimeError("Stockfish process terminated unexpectedly")
            lines.append(line)
            if line.startswith(stop_token):
                break
        return lines

    @staticmethod
    def _parse_info_line(line: str) -> Optional[Dict]:
        """Parse a UCI info line into a dict.

        Returns dict with: multipv, depth, nodes, score_cp, score_mate,
                           wdl_w, wdl_d, wdl_l, pv (list of UCI moves).
        Returns None if the line isn't a valid info line with a PV.
        """
        if not line.startswith("info "):
            return None

        # Skip "info string" lines
        if line.startswith("info string"):
            return None

        tokens = line.split()
        result = {
            "multipv": 1,
            "depth": 0,
            "nodes": 0,
            "score_cp": None,
            "score_mate": None,
            "wdl_w": None,
            "wdl_d": None,
            "wdl_l": None,
            "pv": [],
        }

        i = 1  # skip "info"
        while i < len(tokens):
            tok = tokens[i]

            if tok == "depth" and i + 1 < len(tokens):
                try:
                    result["depth"] = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2

            elif tok == "seldepth" and i + 1 < len(tokens):
                i += 2  # skip

            elif tok == "nodes" and i + 1 < len(tokens):
                try:
                    result["nodes"] = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2

            elif tok == "multipv" and i + 1 < len(tokens):
                try:
                    result["multipv"] = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2

            elif tok == "score" and i + 2 < len(tokens):
                score_type = tokens[i + 1]
                try:
                    score_val = int(tokens[i + 2])
                except ValueError:
                    i += 3
                    continue
                if score_type == "cp":
                    result["score_cp"] = score_val
                elif score_type == "mate":
                    result["score_mate"] = score_val
                i += 3

            elif tok == "wdl" and i + 3 < len(tokens):
                try:
                    w = int(tokens[i + 1])
                    d = int(tokens[i + 2])
                    l_ = int(tokens[i + 3])
                    result["wdl_w"] = w / 1000.0
                    result["wdl_d"] = d / 1000.0
                    result["wdl_l"] = l_ / 1000.0
                except ValueError:
                    pass
                i += 4

            elif tok == "pv":
                result["pv"] = tokens[i + 1:]
                break  # pv is always last

            else:
                i += 1

        # Only return if we got a PV
        if not result["pv"]:
            return None

        return result

    def evaluate_all_legal_moves(self, board: chess.Board) -> List[Dict]:
        """
        Evaluate EVERY legal move via MultiPV search.

        Uses the same approach as DirectLC0Evaluator: sets MultiPV to
        the number of legal moves, runs a single search, collects results.

        Search limits are configurable: depth, nodes, movetime.
        If none are set, defaults to depth 14.

        Returns a list of dicts, one per legal move:
            {
                "move_uci": str,
                "move_san": str,
                "score_cp": int|None,   # centipawns from WHITE's perspective
                "score_mate": int|None,
                "nodes": int,
                "depth": int,
                "wdl_w": float|None,    # 0.0-1.0, White's perspective
                "wdl_d": float|None,
                "wdl_l": float|None,
                "fen_after": str,
            }
        """
        if self._proc is None:
            raise RuntimeError("Engine not started")

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        n_legal = len(legal_moves)

        # Set MultiPV only if it changed
        if self._multipv_set != n_legal:
            self._set_option("MultiPV", str(n_legal))
            self._multipv_set = n_legal

        # Send position
        self._send(f"position fen {board.fen()}")

        # Build go command with configured limits
        go_parts = ["go"]
        if self.max_depth > 0:
            go_parts.append(f"depth {self.max_depth}")
        if self.max_nodes > 0:
            go_parts.append(f"nodes {self.max_nodes}")
        if self.movetime_ms > 0:
            go_parts.append(f"movetime {self.movetime_ms}")
        # Default: depth 14 if nothing else is set
        if len(go_parts) == 1:
            go_parts.append("depth 14")
        self._send(" ".join(go_parts))

        # Pre-compute SAN and fen_after while engine searches
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

        # Collect response
        lines = self._read_until("bestmove")

        # Parse info lines — keep only the LAST info per multipv
        # (Stockfish emits updates at increasing depths)
        pv_infos = {}
        for line in lines:
            parsed = self._parse_info_line(line)
            if parsed is not None:
                pv_infos[parsed["multipv"]] = parsed

        # UCI scores are from STM perspective; convert to White's perspective
        stm_is_white = board.turn == chess.WHITE

        results = []
        seen_moves = set()

        for mpv_idx in sorted(pv_infos.keys()):
            info = pv_infos[mpv_idx]
            pv = info["pv"]
            if not pv:
                continue

            uci = pv[0]
            if uci in seen_moves:
                continue
            seen_moves.add(uci)

            mi = move_info.get(uci, {"san": uci, "fen_after": ""})

            # Convert score from STM to White's perspective
            score_cp = info["score_cp"]
            score_mate = info["score_mate"]
            if not stm_is_white:
                if score_cp is not None:
                    score_cp = -score_cp
                if score_mate is not None:
                    score_mate = -score_mate

            # Convert WDL from STM to White's perspective
            wdl_w = info["wdl_w"]
            wdl_d = info["wdl_d"]
            wdl_l = info["wdl_l"]
            if not stm_is_white and wdl_w is not None:
                wdl_w, wdl_l = wdl_l, wdl_w

            entry = {
                "move_uci": uci,
                "move_san": mi["san"],
                "score_cp": score_cp,
                "score_mate": score_mate,
                "nodes": info["nodes"],
                "depth": info["depth"],
                "wdl_w": wdl_w,
                "wdl_d": wdl_d,
                "wdl_l": wdl_l,
                "fen_after": mi["fen_after"],
            }
            results.append(entry)

        # Fallback for any missed moves
        for move in legal_moves:
            uci = move.uci()
            if uci not in seen_moves:
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
                results.append(entry)

        return results

    def quit(self):
        """Shut down the Stockfish process."""
        if self._proc:
            try:
                self._send("quit")
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def is_alive(self) -> bool:
        """Check if the Stockfish subprocess is still running."""
        return self._proc is not None and self._proc.poll() is None

    def restart(self):
        """Kill and restart the Stockfish engine process."""
        self.quit()
        self.start()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.quit()


class SyncStockfishEvaluator:
    """Drop-in replacement for SyncDirectEvaluator / SyncBatchEvaluator using Stockfish."""

    def __init__(self, stockfish_path: str, threads: int = 1, hash_mb: int = 128,
                 max_depth: int = 0, max_nodes: int = 0, movetime_ms: int = 0,
                 # Ignored LC0 compat args
                 lc0_path: str = "", weights_path: str = "", backend: str = "",
                 batch_size: int = 0, min_nodes: int = 0, nodes_mult: float = 0,
                 max_nodes_lc0: int = 0):
        self._evaluator = DirectStockfishEvaluator(
            stockfish_path=stockfish_path,
            threads=threads,
            hash_mb=hash_mb,
            max_depth=max_depth,
            max_nodes=max_nodes,
            movetime_ms=movetime_ms,
        )

    def start(self):
        self._evaluator.start()

    def evaluate_all_legal_moves(self, board: chess.Board) -> List[Dict]:
        return self._evaluator.evaluate_all_legal_moves(board)

    def quit(self):
        self._evaluator.quit()

    def is_alive(self) -> bool:
        return self._evaluator.is_alive()

    def restart(self):
        self._evaluator.restart()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.quit()
