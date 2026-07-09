"""
DirectLC0Evaluator — Direct UCI subprocess wrapper for LC0.

Bypasses python-chess's engine wrapper to minimize per-call overhead.
Communicates with LC0 via raw stdin/stdout UCI protocol, parsing info
lines with simple string splitting instead of rich object construction.

Drop-in replacement for BatchLC0Evaluator / SyncBatchEvaluator.
Same evaluate_all_legal_moves() return format.

Typical overhead: ~1-2ms per call vs ~8-12ms with python-chess.
"""

import subprocess
import threading
import chess
from typing import List, Dict, Optional


class DirectLC0Evaluator:
    """
    LC0 evaluator using direct UCI subprocess communication.

    Supports both node-based and time-based search limits.
    Parses UCI info lines directly for minimal overhead.
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
        use_exploration_settings: bool = True,
        time_per_move_ms: Optional[float] = None,
    ):
        self.lc0_path = lc0_path
        self.weights_path = weights_path
        self.backend = backend
        self.batch_size = batch_size
        self.threads = threads
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.nodes_mult = nodes_mult
        self.use_exploration_settings = use_exploration_settings
        self.time_per_move_ms = time_per_move_ms
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._multipv_set: Optional[int] = None  # track current MultiPV to avoid resending

    def start(self):
        """Start the LC0 engine process and configure UCI options."""
        self._proc = subprocess.Popen(
            [self.lc0_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Drain stderr in background so it doesn't block
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

        # Wait for UCI handshake
        self._send("uci")
        self._read_until("uciok")

        # Set options
        self._set_option("WeightsFile", self.weights_path)
        self._set_option("Backend", self.backend)
        self._set_option("MinibatchSize", str(self.batch_size))
        self._set_option("Threads", str(self.threads))
        self._set_option("UCI_ShowWDL", "true")

        if self.use_exploration_settings:
            self._set_option("PerPVCounters", "true")
            self._set_option("SmartPruningFactor", "0")
            self._set_option("FpuStrategy", "absolute")
            self._set_option("FpuValue", "0")
            self._set_option("CPuct", "100.0")
            self._set_option("PolicyTemperature", "10.0")

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
        """Read stdout lines until a line starts with stop_token. Returns all lines."""
        lines = []
        while True:
            line = self._proc.stdout.readline().strip()
            if not line and self._proc.poll() is not None:
                raise RuntimeError("LC0 process terminated unexpectedly")
            lines.append(line)
            if line.startswith(stop_token):
                break
        return lines

    def _compute_node_limit(self, n_legal: int) -> int:
        """Compute node budget: clamp(n_legal * nodes_mult, min_nodes, max_nodes)."""
        node_limit = max(self.min_nodes, int(n_legal * self.nodes_mult))
        if self.max_nodes > 0:
            node_limit = min(node_limit, self.max_nodes)
        return node_limit

    @staticmethod
    def _parse_info_line(line: str) -> Optional[Dict]:
        """Parse a UCI info line into a dict.

        Example:
          info depth 1 seldepth 0 time 0 nodes 1 score cp -7413 wdl 1 0 999
               tbhits 0 multipv 1 pv a7a5 e1a1

        Returns dict with: multipv, depth, nodes, score_cp, score_mate,
                           wdl_w, wdl_d, wdl_l, pv (list of UCI moves).
        Returns None if the line isn't a valid info line.
        """
        if not line.startswith("info "):
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
                    l = int(tokens[i + 3])
                    result["wdl_w"] = w / 1000.0
                    result["wdl_d"] = d / 1000.0
                    result["wdl_l"] = l / 1000.0
                except ValueError:
                    pass
                i += 4

            elif tok == "pv":
                result["pv"] = tokens[i + 1:]
                break  # pv is always last

            else:
                i += 1

        # Only return if we got a PV (skip string/bound info lines)
        if not result["pv"]:
            return None

        return result

    def evaluate_all_legal_moves(self, board: chess.Board) -> List[Dict]:
        """
        Evaluate EVERY legal move in ONE engine call via direct UCI.

        Pipelined: sends go command FIRST, then pre-computes SAN/fen_after
        while the GPU evaluates. Overlaps ~1-2ms of CPU work with GPU eval.

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

        # Send position + go command — GPU starts evaluating NOW
        self._send(f"position fen {board.fen()}")

        if self.time_per_move_ms is not None:
            self._send(f"go movetime {int(self.time_per_move_ms)}")
        else:
            node_limit = self._compute_node_limit(n_legal)
            self._send(f"go nodes {node_limit}")

        # Pre-compute SAN and fen_after WHILE GPU evaluates (overlapped)
        # Pipe buffers LC0's info line output (~3.5KB for 35 moves)
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

        # Collect response (GPU output already buffered in pipe)
        lines = self._read_until("bestmove")

        # Parse info lines — keep only the LAST info per multipv
        # (LC0 may emit intermediate updates at increasing depths)
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

        # Fallback for missed moves (shouldn't happen with exploration settings)
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
        """Shut down the LC0 process."""
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
        """Check if the LC0 subprocess is still running."""
        return self._proc is not None and self._proc.poll() is None

    def restart(self):
        """Kill and restart the LC0 engine process."""
        self.quit()
        self.start()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.quit()


class SyncDirectEvaluator:
    """Drop-in replacement for SyncBatchEvaluator using direct UCI."""

    def __init__(self, *args, **kwargs):
        self._evaluator = DirectLC0Evaluator(*args, **kwargs)

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
