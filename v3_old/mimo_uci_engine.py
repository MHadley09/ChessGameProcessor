#!/usr/bin/env python3
"""
mimo_uci_engine.py — UCI-compatible chess engine powered by MIMO V3 model.

Plays chess like a human of specified Elo rating.  Compatible with CuteChess,
Arena, and any UCI GUI.

Usage (standalone):
    python mimo_uci_engine.py \
        --checkpoint checkpoints/v3/best.pt \
        --lc0 path/to/lc0.exe \
        --lc0-weights path/to/weights.pb.gz

    Or omit CLI args and configure paths via UCI setoption after launch.

UCI options:
    EngineElo / OpponentElo — engine and opponent ratings (400-3200, default 1500)
    TimeControl          — Lichess-style string, e.g. "300+3" (default)
    Temperature          — move sampling temperature (default 1.0)
    TopK                 — sample from top K predicted moves (default 20)
    LC0Path / LC0Weights / Checkpoint — engine & model paths
    MaxPossible          — max candidate moves for feature construction (default 218)
"""

import argparse
import math
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch

# Import inference machinery from the companion infer_mimo.py (same directory)
from infer_mimo import (
    MIMOPredictor,
    LC0Engine,
    build_features,
    parse_time_control,
)


# ═══════════════════════════════════════════════════════════════════════════
# UCI Engine
# ═══════════════════════════════════════════════════════════════════════════

class MIMOUCIEngine:
    """Full UCI protocol handler wrapping MIMO inference."""

    ENGINE_NAME = "KRUT-MIMO v3"
    ENGINE_AUTHOR = "Michael Hadley"

    def __init__(self, args):
        self.board = chess.Board()
        self.move_history: List[Tuple[int, int]] = []
        self.predictor: Optional[MIMOPredictor] = None
        self.lc0_engine: Optional[LC0Engine] = None
        self.debug = False

        # UCI options (settable via `setoption name ... value ...`)
        self.options = {
            'EngineElo':     {'type': 'spin',   'default': 1500, 'min': 400, 'max': 3200, 'value': 1500},
            'OpponentElo':   {'type': 'spin',   'default': 1500, 'min': 400, 'max': 3200, 'value': 1500},
            'TimeControl':   {'type': 'string', 'default': '300+3', 'value': '300+3'},
            'Temperature':   {'type': 'string', 'default': '1.0',   'value': '1.0'},
            'TopK':          {'type': 'spin',   'default': 20, 'min': 1, 'max': 220, 'value': 20},
            'LC0Path':       {'type': 'string', 'default': getattr(args, 'lc0', '') or '', 'value': getattr(args, 'lc0', '') or ''},
            'LC0Weights':    {'type': 'string', 'default': getattr(args, 'lc0_weights', '') or '', 'value': getattr(args, 'lc0_weights', '') or ''},
            'Checkpoint':    {'type': 'string', 'default': getattr(args, 'checkpoint', '') or '', 'value': getattr(args, 'checkpoint', '') or ''},
            'MaxPossible':   {'type': 'spin',   'default': 40, 'min': 1, 'max': 220, 'value': 40},
        }
        self._device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # ── Helpers ───────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.debug:
            print(f"info string {msg}", flush=True)

    def _opt(self, name: str):
        return self.options[name]['value']

    def _detect_prev_capture(self) -> float:
        """Check whether the most recent move in the game was a capture."""
        moves = list(self.board.move_stack)
        if not moves:
            return 0.0
        try:
            temp = chess.Board()
            for m in moves[:-1]:
                temp.push(m)
            if temp.is_capture(moves[-1]):
                return 1.0
        except Exception:
            pass
        return 0.0

    # ── UCI command handlers ──────────────────────────────────────────

    def cmd_uci(self):
        print(f"id name {self.ENGINE_NAME}")
        print(f"id author {self.ENGINE_AUTHOR}")
        for name, opt in self.options.items():
            if opt['type'] == 'spin':
                print(f"option name {name} type spin default {opt['default']} "
                      f"min {opt['min']} max {opt['max']}")
            elif opt['type'] == 'check':
                val = 'true' if opt['default'] else 'false'
                print(f"option name {name} type check default {val}")
            else:
                print(f"option name {name} type string default {opt['default']}")
        print("uciok", flush=True)

    def cmd_setoption(self, tokens: List[str]):
        try:
            name_idx = tokens.index('name') + 1
            value_idx = tokens.index('value') + 1
        except ValueError:
            return
        name = ' '.join(tokens[name_idx:value_idx - 1])
        value = ' '.join(tokens[value_idx:])
        if name not in self.options:
            self._log(f"Unknown option: {name}")
            return
        opt = self.options[name]
        if opt['type'] == 'spin':
            opt['value'] = max(opt['min'], min(opt['max'], int(value)))
        elif opt['type'] == 'check':
            opt['value'] = value.lower() == 'true'
        else:
            opt['value'] = value
        self._log(f"Set {name} = {opt['value']}")

    def _ensure_init(self):
        """Init model + LC0 engine if paths are configured and not yet loaded."""
        if self.predictor is None and self._opt('Checkpoint'):
            self.predictor = MIMOPredictor(self._opt('Checkpoint'), self._device)
            self._log(f"Model loaded on {self._device})")
            print(f"info string Model loaded ({self._device})", flush=True)
        if self.lc0_engine is None and self._opt('LC0Path'):
            weights = self._opt('LC0Weights') or None
            self.lc0_engine = LC0Engine(self._opt('LC0Path'), weights)
            self._log("LC0 engine started")
            print("info string LC0 engine started", flush=True)
    def cmd_isready(self):
        self._ensure_init()
        print("readyok", flush=True)

    def cmd_ucinewgame(self):
        self.board = chess.Board()
        self.move_history = []

    def cmd_position(self, tokens: List[str]):
        if not tokens:
            return
        if tokens[0] == 'startpos':
            self.board = chess.Board()
            moves_start = 2 if len(tokens) > 1 and tokens[1] == 'moves' else len(tokens)
        elif tokens[0] == 'fen':
            fen_parts = []
            i = 1
            while i < len(tokens) and tokens[i] != 'moves':
                fen_parts.append(tokens[i])
                i += 1
            self.board = chess.Board(' '.join(fen_parts))
            moves_start = i + 1 if i < len(tokens) else len(tokens)
        else:
            return

        self.move_history = []
        for uci_str in tokens[moves_start:]:
            try:
                move = chess.Move.from_uci(uci_str)
                self.move_history.append((move.from_square, move.to_square))
                self.board.push(move)
            except Exception:
                pass

    def cmd_go(self, tokens: List[str]):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            print("bestmove 0000", flush=True)
            return

        # Fallback if model/engine not loaded
        if not self.predictor or not self.lc0_engine:
            self._log("Model or LC0 not loaded — picking random move")
            print(f"bestmove {random.choice(legal_moves).uci()}", flush=True)
            return

        # ── Parse go params ──────────────────────────────────────────
        params: Dict = {}
        i = 0
        while i < len(tokens):
            key = tokens[i]
            if key in ('wtime', 'btime', 'winc', 'binc', 'movetime',
                       'movestogo', 'depth', 'nodes'):
                if i + 1 < len(tokens):
                    try:
                        params[key] = int(tokens[i + 1])
                    except ValueError:
                        pass
                i += 2
            elif key == 'infinite':
                params['infinite'] = True
                i += 1
            else:
                i += 1

        # ── Clock time for features ──────────────────────────────────
        color = 'White' if self.board.turn == chess.WHITE else 'Black'
        if color == 'White' and 'wtime' in params:
            clock_time = params['wtime'] / 1000.0
        elif color == 'Black' and 'btime' in params:
            clock_time = params['btime'] / 1000.0
        else:
            initial, inc = parse_time_control(self._opt('TimeControl'))
            mn = self.board.fullmove_number
            clock_time = max(0.0, initial - mn * 5 + inc * mn)

        # ── Elo mapping: engine plays the side to move ──────────────
        if self.board.turn == chess.WHITE:
            white_elo = self._opt('EngineElo')
            black_elo = self._opt('OpponentElo')
        else:
            white_elo = self._opt('OpponentElo')
            black_elo = self._opt('EngineElo')

        game_meta = {
            'white_elo': white_elo,
            'black_elo': black_elo,
            'time_control': self._opt('TimeControl'),
            'clock_time': clock_time,
            'move_no': self.board.fullmove_number,
            'prev_capture': self._detect_prev_capture(),
        }

        # ── LC0 evaluation ───────────────────────────────────────────
        t0 = time.time()
        evals = self.lc0_engine.evaluate(self.board)
        lc0_ms = (time.time() - t0) * 1000

        if not evals:
            self._log("LC0 returned no evaluations — random move")
            print(f"bestmove {random.choice(legal_moves).uci()}", flush=True)
            return

        # ── Build features ───────────────────────────────────────────
        history_pairs = self.move_history[-2:] if self.move_history else None
        features = build_features(
            self.board, history_pairs, evals, game_meta,
            self._opt('MaxPossible'),
            model_max_possible=self.predictor.max_possible,
        )

        # ── MIMO inference ───────────────────────────────────────────
        t0 = time.time()
        preds = self.predictor.predict(features)
        infer_ms = (time.time() - t0) * 1000

        move_ucis = features['_move_ucis']
        probs = preds['move_probs'][:len(move_ucis)]

        # ── Move selection (always sample from top-K) ────────────────
        temperature = float(self._opt('Temperature'))
        top_k = min(self._opt('TopK'), len(probs))

        ranked = np.argsort(probs)[::-1][:top_k]
        top_probs = probs[ranked].astype(np.float64)
        log_p = np.log(top_probs + 1e-10)
        scaled = np.exp(log_p / max(temperature, 0.01))
        scaled /= scaled.sum()
        best_idx = int(ranked[np.random.choice(len(ranked), p=scaled)])

        best_move = move_ucis[best_idx]
        best_prob = float(probs[best_idx])

        # ── UCI info output (before think-time delay) ────────────────
        info_parts = []
        if 'wdl_before' in preds:
            wdl = preds['wdl_before']
            w_pm = int(wdl[0] * 1000)
            d_pm = int(wdl[1] * 1000)
            l_pm = max(0, 1000 - w_pm - d_pm)
            cp_approx = int((wdl[0] - wdl[2]) * 400)
            info_parts.append(f"info depth 1 score cp {cp_approx} wdl {w_pm} {d_pm} {l_pm} "
                              f"pv {best_move} "
                              f"string prob={best_prob:.1%} "
                              f"mistake={preds['mistake_prob']:.1%} "
                              f"time_pred={preds['predicted_time_s']:.1f}s "
                              f"lc0={lc0_ms:.0f}ms mimo={infer_ms:.0f}ms")
        for line in info_parts:
            print(line, flush=True)

        # ── Human-like think time (predicted time + noise) ───────────
        predicted_time = preds['predicted_time_s']
        noise_factor = 1.0 + random.uniform(-0.2, 0.2)  # ±20% jitter
        think_time = max(0.5, predicted_time * noise_factor)  # minimum 0.5s

        # Never exceed available clock time (leave 1s safety buffer)
        if color == 'White' and 'wtime' in params:
            max_think = params['wtime'] / 1000.0 - 1.0
        elif color == 'Black' and 'btime' in params:
            max_think = params['btime'] / 1000.0 - 1.0
        else:
            max_think = 30.0  # fallback cap

        think_time = min(think_time, max(0.1, max_think))

        # Subtract time already spent on LC0 + inference
        elapsed = (lc0_ms + infer_ms) / 1000.0
        remaining_delay = max(0.0, think_time - elapsed)
        if remaining_delay > 0:
            time.sleep(remaining_delay)

        print(f"bestmove {best_move}", flush=True)

    def cmd_debug(self, tokens: List[str]):
        self.debug = bool(tokens) and tokens[0] == 'on'

    def cmd_quit(self):
        if self.lc0_engine:
            self.lc0_engine.close()

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self):
        """UCI stdin/stdout loop."""
        while True:
            try:
                line = input().strip()
            except EOFError:
                break
            if not line:
                continue

            tokens = line.split()
            cmd = tokens[0]

            if cmd == 'uci':
                self.cmd_uci()
            elif cmd == 'debug':
                self.cmd_debug(tokens[1:])
            elif cmd == 'isready':
                self.cmd_isready()
            elif cmd == 'setoption':
                self.cmd_setoption(tokens)
            elif cmd == 'ucinewgame':
                self.cmd_ucinewgame()
            elif cmd == 'position':
                self.cmd_position(tokens[1:])
            elif cmd == 'go':
                self.cmd_go(tokens[1:])
            elif cmd == 'stop':
                pass  # MIMO inference is synchronous — nothing to stop
            elif cmd == 'quit':
                self.cmd_quit()
                break

            sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='KRUT-MIMO v3 — UCI chess engine powered by human-move prediction')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to MIMO V3 checkpoint (.pt)')
    parser.add_argument('--lc0', type=str, default='',
                        help='Path to LC0 executable')
    parser.add_argument('--lc0-weights', type=str, default='',
                        help='Path to LC0 network weights (.pb.gz)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    engine = MIMOUCIEngine(args)
    if args.checkpoint and args.lc0:
        engine._ensure_init()
        if engine.predictor is None:
            print("info string ERROR: Failed to load model checkpoint", flush=True)
        if engine.lc0_engine is None:
            print("info string ERROR: Failed to start LC0 engine", flush=True)

    engine.run()


if __name__ == '__main__':
    main()
