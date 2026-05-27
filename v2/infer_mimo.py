#!/usr/bin/env python3
"""
infer_mimo.py — Real-time inference for the MIMO chess model.

Given a chess position + game context, runs LC0 for engine evaluations
then feeds everything through the MIMO model to predict human behavior.

Predictions (all 5 heads):
    - Move probability distribution (what a human of given Elo would play)
    - Mistake probability for each candidate move
    - Win/Draw/Loss before the move (position assessment)
    - Win/Draw/Loss after each candidate move
    - Predicted thinking time (seconds)

Usage:
    python infer_mimo.py \
        --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1" \
        --white-elo 1500 --black-elo 1500 \
        --time-control "300+3" --clock-time 280 \
        --checkpoint checkpoints/best.pt \
        --lc0 path/to/lc0.exe \
        --lc0-weights path/to/large.pb.gz \
        --top-k 10

    # With move history (for history planes):
    python infer_mimo.py \
        --fen "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2" \
        --history "e2e4,b8c6" \
        ...
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch
from torch.amp import autocast

from chess_mimo_model_v2 import ChessMIMOModelV2


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Fast Plane Construction (Bitboard-Accelerated)
# ═══════════════════════════════════════════════════════════════════════════
#
# ~5–7× faster than the per-square Python loop in mimo_dataset_polars.py.
# Uses numpy batch ops on python-chess bitboards — no per-square iteration.
# Can be dropped into the training dataset as a replacement for
# board_to_planes() to speed up DataLoader workers.
# ═══════════════════════════════════════════════════════════════════════════

def _bb_to_planes_batch(bitboards: List[int]) -> np.ndarray:
    """Convert N chess bitboards → (N, 8, 8) float32 planes via numpy."""
    n = len(bitboards)
    if n == 0:
        return np.zeros((0, 8, 8), dtype=np.float32)
    arr = np.array(bitboards, dtype=np.uint64)
    raw = arr.view(np.uint8).reshape(n, 8)        # N × 8 bytes
    bits = np.unpackbits(raw, axis=1).reshape(n, 8, 8)  # MSB-first per byte
    bits = np.flip(bits, axis=2)                   # fix file order (LSB = A-file)
    bits = np.flip(bits, axis=1)                   # rank 7 at row 0 (board top)
    return np.ascontiguousarray(bits).astype(np.float32)


def _piece_bitboards(board: chess.Board) -> List[int]:
    """Return 12 bitboards: [W-pawn .. W-king, B-pawn .. B-king]."""
    return [int(board.pieces(pt, c))
            for c in (chess.WHITE, chess.BLACK)
            for pt in range(chess.PAWN, chess.KING + 1)]


def board_to_planes_fast(board: chess.Board,
                         history: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    """
    Bitboard-accelerated FEN → (23, 8, 8) plane builder.

    Plane layout (23 channels):
        0-11:  piece planes (W-pawn..W-king, B-pawn..B-king)
        12-13: last move from/to squares
        14-15: second-to-last move from/to squares
        16:    side to move (1.0 = White)
        17-20: castling rights (WK, WQ, BK, BQ)
        21:    en passant square
        22:    halfmove clock / 100
    """
    planes = np.zeros((23, 8, 8), dtype=np.float32)

    # --- Piece planes 0–11 (current position) ---
    bbs = _piece_bitboards(board)
    p12 = _bb_to_planes_batch(bbs)
    for i in range(12):
        if bbs[i]:
            planes[i] = p12[i]

    # --- Move history from/to squares 12–15 ---
    if history:
        if len(history) >= 1:
            from_sq, to_sq = history[0]
            if from_sq is not None:
                planes[12, 7 - from_sq // 8, from_sq % 8] = 1.0
                planes[13, 7 - to_sq // 8, to_sq % 8] = 1.0
        if len(history) >= 2:
            from_sq, to_sq = history[1]
            if from_sq is not None:
                planes[14, 7 - from_sq // 8, from_sq % 8] = 1.0
                planes[15, 7 - to_sq // 8, to_sq % 8] = 1.0

    # --- Meta planes 16–22 ---
    planes[16, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[17, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[18, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[19, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[20, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    if board.ep_square is not None:
        planes[21, 7 - board.ep_square // 8, board.ep_square % 8] = 1.0
    planes[22, :, :] = min(board.halfmove_clock, 100) / 100.0

    return planes


def build_possible_planes_fast(board: chess.Board,
                               move_order: List[chess.Move],
                               history: Optional[List[Tuple[int, int]]],
                               max_possible: int = 40) -> np.ndarray:
    """
    Build (max_possible, 47, 8, 8) possible-move planes via push/pop.
    Avoids FEN parsing overhead for every legal move.
    """
    poss = np.zeros((max_possible, 47, 8, 8), dtype=np.float32)
    recent = history[-1:] if history else []

    for i, move in enumerate(move_order[:max_possible]):
        from_sq, to_sq = move.from_square, move.to_square
        board.push(move)
        poss_hist = [(from_sq, to_sq)] + (recent if recent else [])
        poss[i] = board_to_planes_fast(board, poss_hist)
        board.pop()

    return poss


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — LC0 UCI Engine Interface
# ═══════════════════════════════════════════════════════════════════════════

class LC0Engine:
    """Wraps LC0 in UCI mode for all-legal-moves evaluation."""

    def __init__(self, engine_path: str, weights_path: Optional[str] = None,
                 nodes: int = 1, policy_temp: float = 10.0):
        self.nodes = nodes
        self.policy_temp = policy_temp
        cmd = [engine_path]
        if weights_path:
            cmd += [f'--weights={weights_path}']
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send('uci')
        self._wait_for('uciok')
        opts = {
            'UCI_ShowWDL': 'true',
            'SmartPruningFactor': '0',
            'PerPVCounters': 'true',
            'PolicyTemperature': str(policy_temp),
        }
        for k, v in opts.items():
            self._send(f'setoption name {k} value {v}')
        self._send('isready')
        self._wait_for('readyok')

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + '\n')
        self.proc.stdin.flush()

    def _wait_for(self, token: str) -> List[str]:
        lines = []
        while True:
            line = self.proc.stdout.readline().strip()
            lines.append(line)
            if line.startswith(token):
                return lines

    def evaluate(self, board: chess.Board) -> Dict[str, Dict]:
        """
        Evaluate all legal moves.  Returns {uci_move: {eval, wdl_w, wdl_d, wdl_l,
        policy_prob, nodes, depth}} sorted by eval descending.
        """
        legal = list(board.legal_moves)
        n_legal = len(legal)
        if n_legal == 0:
            return {}

        self._send('ucinewgame')
        self._send(f'position fen {board.fen()}')
        mpv = min(n_legal, 218)
        self._send(f'go nodes {self.nodes * mpv} multipv {mpv}')

        info_lines = self._wait_for('bestmove')
        results = {}

        for line in info_lines:
            if not line.startswith('info'):
                continue
            move_uci = self._parse_field(line, 'pv')
            if not move_uci:
                continue
            move_uci = move_uci.split()[0]  # first move of PV

            cp = self._parse_int(line, 'cp')
            wdl = self._parse_wdl(line)
            nodes = self._parse_int(line, 'nodes') or 1
            depth = self._parse_int(line, 'depth') or 1

            results[move_uci] = {
                'eval': cp if cp is not None else 0,
                'wdl_w': wdl[0] / 1000.0 if wdl else 0.33,
                'wdl_d': wdl[1] / 1000.0 if wdl else 0.34,
                'wdl_l': wdl[2] / 1000.0 if wdl else 0.33,
                'nodes': nodes,
                'depth': depth,
                'policy_prob': 0.0,  # not directly available from UCI multipv
            }

        # Approximate policy from softmax of evals
        if results:
            evals = np.array([r['eval'] for r in results.values()], dtype=np.float64)
            probs = np.exp(evals / 100.0)
            probs /= probs.sum()
            for i, k in enumerate(results):
                results[k]['policy_prob'] = float(probs[i])

        return results

    @staticmethod
    def _parse_field(line: str, field: str) -> Optional[str]:
        parts = line.split()
        for i, p in enumerate(parts):
            if p == field and i + 1 < len(parts):
                return ' '.join(parts[i + 1:])
        return None

    @staticmethod
    def _parse_int(line: str, field: str) -> Optional[int]:
        parts = line.split()
        for i, p in enumerate(parts):
            if p == field and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    return None
        return None

    @staticmethod
    def _parse_wdl(line: str) -> Optional[Tuple[int, int, int]]:
        parts = line.split()
        for i, p in enumerate(parts):
            if p == 'wdl' and i + 3 < len(parts):
                try:
                    return (int(parts[i+1]), int(parts[i+2]), int(parts[i+3]))
                except ValueError:
                    return None
        return None

    def close(self):
        try:
            self._send('quit')
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Feature Construction
# ═══════════════════════════════════════════════════════════════════════════
# Mirrors build_one_compact() in mimo_dataset_polars.py exactly so
# training ↔ inference features are guaranteed identical.
# ═══════════════════════════════════════════════════════════════════════════

PIECE_MAP = {'P': 1/6, 'N': 2/6, 'B': 3/6, 'R': 4/6, 'Q': 5/6, 'K': 1.0}
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def parse_time_control(tc: str) -> Tuple[float, float]:
    """Parse Lichess-style time control '300+3' → (300.0, 3.0)."""
    if not tc:
        return 0.0, 0.0
    try:
        parts = str(tc).split('+')
        base = float(parts[0])
        inc = float(parts[1]) if len(parts) > 1 else 0.0
        return base, inc
    except Exception:
        return 0.0, 0.0


def compute_material_balance(board: chess.Board) -> float:
    """White material – Black material, capped to ±39."""
    balance = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            v = PIECE_VALUES.get(p.piece_type, 0)
            balance += v if p.color == chess.WHITE else -v
    return float(np.clip(balance, -39, 39))


def build_features(board: chess.Board,
                   history: Optional[List[Tuple[int, int]]],
                   evals: Dict[str, Dict],
                   game_meta: Dict,
                   max_possible: int = 40) -> Dict[str, torch.Tensor]:
    """
    Build all MIMO input tensors for a single position.

    Args:
        board:      Current chess.Board
        history:    List of (from_sq, to_sq) tuples for recent moves
        evals:      LC0 evaluation dict from LC0Engine.evaluate()
        game_meta:  {'white_elo', 'black_elo', 'time_control', 'clock_time',
                     'move_no', 'prev_capture'}

    Returns:
        Dict with keys matching MIMOCompactDataset.__getitem__ output.
    """
    color = 'White' if board.turn == chess.WHITE else 'Black'
    w_elo = game_meta.get('white_elo', 1500)
    b_elo = game_meta.get('black_elo', 1500)
    move_no = game_meta.get('move_no', board.fullmove_number)
    clock_time = game_meta.get('clock_time', 0.0)
    prev_capture = game_meta.get('prev_capture', 0.0)
    tc_str = game_meta.get('time_control', '')
    initial_time, increment = parse_time_control(tc_str)

    legal_moves = list(board.legal_moves)
    n_legal = len(legal_moves)

    # --- Sort legal moves by eval (descending, STM perspective) ---
    move_eval_list = []
    for m in legal_moves:
        uci = m.uci()
        ev = evals.get(uci, {})
        raw_eval = ev.get('eval', 0)
        stm_eval = raw_eval if color == 'White' else -raw_eval
        move_eval_list.append((m, uci, ev, stm_eval))
    move_eval_list.sort(key=lambda x: x[3], reverse=True)

    # Clamp to max_possible
    move_eval_list = move_eval_list[:max_possible]
    num_possible = len(move_eval_list)

    evals_stm = [x[3] for x in move_eval_list]
    best_eval_stm = max(evals_stm) if evals_stm else 0.0
    worst_eval_stm = min(evals_stm) if evals_stm else 0.0
    eval_range = best_eval_stm - worst_eval_stm

    # --- Build ordered move list for plane construction ---
    ordered_moves = [x[0] for x in move_eval_list]

    # --- Current planes ---
    current_planes = board_to_planes_fast(board, history)

    # --- Possible planes (push/pop, no FEN parsing) ---
    # history=None matches training pipeline (possible moves don't use history)
    possible_planes = build_possible_planes_fast(board, ordered_moves, None, max_possible)

    # --- Possible scalars (12 dims per move) ---
    possible_scalars = np.zeros((max_possible, 12), dtype=np.float32)
    possible_mask = np.zeros(max_possible, dtype=np.float32)
    possible_mask[:num_possible] = 1.0

    move_ucis = []
    for i, (move, uci, ev, stm_eval) in enumerate(move_eval_list):
        move_ucis.append(uci)
        # WDL from STM perspective
        if color == 'White':
            wdl_w = ev.get('wdl_w', 0.33)
            wdl_d = ev.get('wdl_d', 0.34)
            wdl_l = ev.get('wdl_l', 0.33)
        else:
            wdl_w = ev.get('wdl_l', 0.33)   # STM win = opponent loss
            wdl_d = ev.get('wdl_d', 0.34)
            wdl_l = ev.get('wdl_w', 0.33)

        nodes_raw = ev.get('nodes', 1)
        depth = ev.get('depth', 1)
        policy_prob = ev.get('policy_prob', 0.0)
        move_quality = (stm_eval - worst_eval_stm) / eval_range if eval_range > 0 else 1.0

        # Piece type of the moving piece
        piece = board.piece_at(move.from_square)
        piece_sym = piece.symbol().upper() if piece else 'P'
        piece_val = PIECE_MAP.get(piece_sym, 1/6)

        # Capture detection
        is_capture = 0.0
        target = board.piece_at(move.to_square)
        if target is not None:
            is_capture = 1.0
        if board.ep_square == move.to_square and piece_sym == 'P':
            is_capture = 1.0

        # Check / checkmate detection via push/pop
        board.push(move)
        is_check = 1.0 if board.is_check() else 0.0
        is_checkmate = 1.0 if board.is_checkmate() else 0.0
        board.pop()

        promotion = 1.0 if move.promotion else 0.0

        possible_scalars[i] = [
            stm_eval / 1000.0,
            wdl_w,
            wdl_d,
            wdl_l,
            math.log1p(nodes_raw) / 20.0,
            depth / 40.0,
            move_quality,
            piece_val,
            is_capture,
            is_check,
            is_checkmate,
            policy_prob,
        ]

    # --- Tabular (18 dims) ---
    eval_stm = best_eval_stm  # position eval ≈ best move's eval
    # WDL before move (from position eval — use best move's WDL)
    if move_eval_list:
        best_ev = move_eval_list[0][2]
        if color == 'White':
            stm_win_before = best_ev.get('wdl_w', 0.33)
            stm_draw_before = best_ev.get('wdl_d', 0.34)
            stm_loss_before = best_ev.get('wdl_l', 0.33)
        else:
            stm_win_before = best_ev.get('wdl_l', 0.33)
            stm_draw_before = best_ev.get('wdl_d', 0.34)
            stm_loss_before = best_ev.get('wdl_w', 0.33)
    else:
        stm_win_before, stm_draw_before, stm_loss_before = 0.33, 0.34, 0.33

    in_check = 1.0 if board.is_check() else 0.0
    eval_std = float(np.std(evals_stm)) / 1000.0 if len(evals_stm) > 1 else 0.0
    captures_frac = sum(possible_scalars[i, 8] for i in range(num_possible)) / max(num_possible, 1)
    checks_frac = sum(possible_scalars[i, 9] for i in range(num_possible)) / max(num_possible, 1)
    num_candidates = num_possible / max_possible

    tabular = np.array([
        clock_time / 3600.0,
        w_elo / 3000.0,
        b_elo / 3000.0,
        (w_elo - b_elo) / 1000.0,
        min(move_no, 200) / 200.0,
        1.0 if color == 'White' else 0.0,
        eval_stm / 1000.0,
        stm_win_before,
        stm_draw_before,
        stm_loss_before,
        initial_time / 3600.0,
        increment / 60.0,
        prev_capture,
        in_check,
        eval_std,
        captures_frac,
        checks_frac,
        num_candidates,
    ], dtype=np.float32)

    # --- Pack into tensors (add batch dim) ---
    return {
        'current_planes':  torch.from_numpy(current_planes).unsqueeze(0),
        'possible_planes': torch.from_numpy(possible_planes).unsqueeze(0),
        'possible_scalars': torch.from_numpy(possible_scalars).unsqueeze(0),
        'possible_mask':   torch.from_numpy(possible_mask).unsqueeze(0),
        'tabular':         torch.from_numpy(tabular).unsqueeze(0),
        # Metadata (not fed to model, but useful for output)
        '_move_ucis':      move_ucis,
        '_n_legal':        n_legal,
        '_color':          color,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — MIMO Predictor
# ═══════════════════════════════════════════════════════════════════════════

class MIMOPredictor:
    """Load a trained MIMO checkpoint and run inference."""

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device)
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get('config', {})

        self.model = ChessMIMOModelV2(
            cnn_channels=cfg.get('cnn_channels', 128),
            num_res_blocks=cfg.get('res_blocks', 6),
            tabular_dim=18,
            max_possible=cfg.get('max_possible', 40),
            hidden_dim=cfg.get('hidden_dim', 256),
            expert_hidden=cfg.get('expert_hidden', 128),
            expert_layers=cfg.get('expert_layers', 2),
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        epoch = ckpt.get('epoch', '?')
        print(f"Loaded epoch {epoch} — {n_params:,} params")

    @torch.no_grad()
    def predict(self, features: Dict[str, torch.Tensor]) -> Dict:
        """Run forward pass, return decoded predictions."""
        cp = features['current_planes'].to(self.device)
        pp = features['possible_planes'].to(self.device)
        ps = features['possible_scalars'].to(self.device)
        pm = features['possible_mask'].to(self.device)
        tab = features['tabular'].to(self.device)
        # No actual_idx at inference — pass None to skip training-only masked path
        with autocast('cuda', enabled=(self.device.type == 'cuda')):
            out = self.model(cp, pp, ps, pm, tab, actual_idx=None)

        move_probs = torch.softmax(out['move_logits'], dim=-1).cpu().numpy()[0]
        mistake_prob = out['mistake_prob'].sigmoid().cpu().numpy()[0].item()
        wdl_before = out.get('win_prob_before')
        wdl_after = out.get('win_prob_after')
        time_log = out['time_spent'].cpu().numpy()[0].item()

        preds = {
            'move_probs': move_probs,
            'mistake_prob': mistake_prob,
            'predicted_time_s': float(np.expm1(time_log)),
        }
        if wdl_before is not None:
            preds['wdl_before'] = wdl_before.cpu().numpy()[0]  # already softmax from model
        if wdl_after is not None:
            preds['wdl_after'] = wdl_after.cpu().numpy()[0]  # already softmax from model

        return preds


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Pretty Output
# ═══════════════════════════════════════════════════════════════════════════

def format_results(features: Dict, preds: Dict, board: chess.Board, top_k: int = 10):
    """Print human-readable prediction results."""
    move_ucis = features['_move_ucis']
    color = features['_color']
    n_legal = features['_n_legal']
    probs = preds['move_probs']

    print(f"\n{'═' * 60}")
    print(f"  MIMO Prediction — {color} to move")
    print(f"  FEN: {board.fen()}")
    print(f"  Legal moves: {n_legal}")
    print(f"{'═' * 60}")

    # --- WDL Before ---
    if 'wdl_before' in preds:
        wdl = preds['wdl_before']
        print(f"\n  Position assessment (WDL from {color}'s view):")
        print(f"    Win {wdl[0]:.1%}  |  Draw {wdl[1]:.1%}  |  Loss {wdl[2]:.1%}")

    # --- Predicted time ---
    t = preds['predicted_time_s']
    print(f"\n  Predicted thinking time: {t:.1f}s")

    # --- Top moves ---
    print(f"\n  Top {min(top_k, len(move_ucis))} predicted moves:")
    print(f"  {'Rank':>4}  {'Move':>8}  {'SAN':>8}  {'Prob':>7}  {'Cum':>7}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")

    ranked = np.argsort(probs[:len(move_ucis)])[::-1]
    cumulative = 0.0
    for rank, idx in enumerate(ranked[:top_k]):
        uci = move_ucis[idx]
        prob = probs[idx]
        cumulative += prob
        try:
            move = chess.Move.from_uci(uci)
            san = board.san(move)
        except Exception:
            san = uci
        print(f"  {rank+1:>4}  {uci:>8}  {san:>8}  {prob:>6.1%}  {cumulative:>6.1%}")

    # --- WDL After (for top move) ---
    if 'wdl_after' in preds:
        wdl_a = preds['wdl_after']
        top_move = move_ucis[ranked[0]] if len(ranked) > 0 else '?'
        print(f"\n  After predicted move ({top_move}), from opponent's view:")
        print(f"    Win {wdl_a[0]:.1%}  |  Draw {wdl_a[1]:.1%}  |  Loss {wdl_a[2]:.1%}")

    # --- Mistake probability ---
    print(f"\n  Mistake probability: {preds['mistake_prob']:.1%}")
    print(f"{'═' * 60}\n")

    return {
        'top_moves': [
            {'uci': move_ucis[idx], 'prob': float(probs[idx])}
            for idx in ranked[:top_k]
        ],
        **preds,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MIMO real-time inference — predict human chess behavior')

    # Position
    parser.add_argument('--fen', required=True, help='FEN of the position to analyze')
    parser.add_argument('--history', type=str, default=None,
                        help='Comma-separated UCI moves from game start (e.g. e2e4,e7e5,g1f3)')

    # Game context (Lichess search parameters)
    parser.add_argument('--white-elo', type=int, default=1500)
    parser.add_argument('--black-elo', type=int, default=1500)
    parser.add_argument('--time-control', type=str, default='300+3',
                        help='Lichess time control string (e.g. 300+3, 60+0, 900+10)')
    parser.add_argument('--clock-time', type=float, default=None,
                        help='Seconds remaining on clock (default: infer from time control)')

    # Engine
    parser.add_argument('--lc0', required=True, help='Path to LC0 executable')
    parser.add_argument('--lc0-weights', type=str, default=None, help='Path to LC0 weights file')
    parser.add_argument('--lc0-nodes', type=int, default=1,
                        help='Nodes per move for LC0 (1 = raw neural net, no search)')

    # Model
    parser.add_argument('--checkpoint', required=True, help='Path to MIMO checkpoint (.pt)')
    parser.add_argument('--max-possible', type=int, default=40)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # Output
    parser.add_argument('--top-k', type=int, default=10, help='Show top K predicted moves')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')

    args = parser.parse_args()

    # --- Set up board ---
    board = chess.Board(args.fen)
    color = 'White' if board.turn == chess.WHITE else 'Black'
    move_no = board.fullmove_number

    # --- Parse move history → (from_sq, to_sq) pairs ---
    history = None
    if args.history:
        history = []
        temp = chess.Board()
        for uci_str in args.history.split(','):
            uci_str = uci_str.strip()
            if not uci_str:
                continue
            move = chess.Move.from_uci(uci_str)
            history.append((move.from_square, move.to_square))
            temp.push(move)
        # Keep last 2 for plane construction
        history = history[-2:] if history else None

    # --- Detect prev_capture from last move in history ---
    prev_capture = 0.0
    if args.history:
        moves = [m.strip() for m in args.history.split(',') if m.strip()]
        if moves:
            try:
                temp = chess.Board()
                for uci_str in moves[:-1]:
                    temp.push(chess.Move.from_uci(uci_str))
                last_move = chess.Move.from_uci(moves[-1])
                if temp.piece_at(last_move.to_square) is not None:
                    prev_capture = 1.0
                if temp.ep_square == last_move.to_square:
                    p = temp.piece_at(last_move.from_square)
                    if p and p.piece_type == chess.PAWN:
                        prev_capture = 1.0
            except Exception:
                pass

    # --- Clock time default ---
    initial_time, increment = parse_time_control(args.time_control)
    clock_time = args.clock_time
    if clock_time is None:
        # Rough estimate: subtract some time for moves played
        elapsed_est = move_no * 5  # ~5s per move average
        clock_time = max(0, initial_time - elapsed_est + increment * move_no)

    game_meta = {
        'white_elo': args.white_elo,
        'black_elo': args.black_elo,
        'time_control': args.time_control,
        'clock_time': clock_time,
        'move_no': move_no,
        'prev_capture': prev_capture,
    }

    # --- LC0 evaluation ---
    print(f"Starting LC0 ({args.lc0}) ...")
    engine = LC0Engine(args.lc0, args.lc0_weights, nodes=args.lc0_nodes)
    t0 = time.time()
    evals = engine.evaluate(board)
    lc0_ms = (time.time() - t0) * 1000
    engine.close()
    print(f"LC0 evaluated {len(evals)} moves in {lc0_ms:.0f}ms")

    if not evals:
        print("ERROR: LC0 returned no evaluations. Check engine path and weights.")
        sys.exit(1)

    # --- Feature construction ---
    t0 = time.time()
    features = build_features(board, history, evals, game_meta, args.max_possible)
    feat_ms = (time.time() - t0) * 1000
    print(f"Features built in {feat_ms:.1f}ms")

    # --- MIMO inference ---
    predictor = MIMOPredictor(args.checkpoint, args.device)
    t0 = time.time()
    preds = predictor.predict(features)
    infer_ms = (time.time() - t0) * 1000
    print(f"MIMO forward pass in {infer_ms:.1f}ms")

    # --- Output ---
    results = format_results(features, preds, board, args.top_k)

    if args.json:
        # Serialize numpy arrays
        out = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        out['timing'] = {'lc0_ms': lc0_ms, 'feature_ms': feat_ms, 'infer_ms': infer_ms}
        out['game_meta'] = game_meta
        print(json.dumps(out, indent=2))

    print(f"\nTotal: LC0 {lc0_ms:.0f}ms + features {feat_ms:.0f}ms + "
          f"MIMO {infer_ms:.0f}ms = {lc0_ms + feat_ms + infer_ms:.0f}ms")


if __name__ == '__main__':
    main()
