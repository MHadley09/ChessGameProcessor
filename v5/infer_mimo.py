#!/usr/bin/env python3
"""
infer_mimo.py — Real-time inference for the MIMO V5 chess model.

Given a chess position + game context, runs LC0 for engine evaluations
then feeds everything through the MIMO model to predict human behavior.

Predictions (4 base heads + V5 extras):
    - Move probability distribution (what a human of given Elo would play)
    - Mistake probability for each candidate move
    - Win/Draw/Loss before the move (position assessment, from White's perspective)
    - Predicted thinking time (seconds)
    - Contrastive embeddings (V5, internal — used for preference modeling)
    - Phase weights (V5, if phase-gated experts enabled)

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
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine
import numpy as np
import torch
from torch.amp import autocast

from chess_mimo_model_v5 import ChessMIMOModelV5


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Fast Plane Construction (Bitboard-Accelerated)
# ═══════════════════════════════════════════════════════════════════════════

def _bb_to_planes_batch(bitboards: List[int]) -> np.ndarray:
    """Convert N chess bitboards → (N, 8, 8) float32 planes via numpy."""
    n = len(bitboards)
    if n == 0:
        return np.zeros((0, 8, 8), dtype=np.float32)
    arr = np.array(bitboards, dtype=np.uint64)
    raw = arr.view(np.uint8).reshape(n, 8)
    bits = np.unpackbits(raw, axis=1).reshape(n, 8, 8)
    bits = np.flip(bits, axis=2)
    bits = np.flip(bits, axis=1)
    return np.ascontiguousarray(bits).astype(np.float32)


def _piece_bitboards(board: chess.Board) -> List[int]:
    """Return 12 bitboards: [W-pawn .. W-king, B-pawn .. B-king]."""
    return [int(board.pieces(pt, c))
            for c in (chess.WHITE, chess.BLACK)
            for pt in range(chess.PAWN, chess.KING + 1)]



def compute_game_phase_from_board(board, ply: int) -> int:
    """Heuristic game phase anchor for PhaseEncoder / FiLM conditioning.

    These are NOT ground-truth labels — they provide initial anchoring so
    the PhaseEncoder and FiLM conditioner have a reasonable starting signal.
    The model learns the real phase boundaries through training.

    Anchoring rules:
        Opening (0):    ply <= 20 (~first 10 full moves)
        Endgame (2):    material heuristic — no queens and each side has
                        at most rook + minor, or queen(s) with no other pieces
        Middlegame (1): everything else

    Args:
        board: chess.Board instance
        ply: half-move count from game start
    """
    # --- Opening: first ~10 full moves ---
    if ply <= 20:
        return 0

    # --- Endgame: material-based ---
    import chess
    w_queen = len(board.pieces(chess.QUEEN, chess.WHITE))
    b_queen = len(board.pieces(chess.QUEEN, chess.BLACK))
    w_rook  = len(board.pieces(chess.ROOK, chess.WHITE))
    b_rook  = len(board.pieces(chess.ROOK, chess.BLACK))
    w_minor = (len(board.pieces(chess.BISHOP, chess.WHITE))
             + len(board.pieces(chess.KNIGHT, chess.WHITE)))
    b_minor = (len(board.pieces(chess.BISHOP, chess.BLACK))
             + len(board.pieces(chess.KNIGHT, chess.BLACK)))

    # No queens: endgame if each side has at most a rook + minor piece
    if w_queen == 0 and b_queen == 0:
        if (w_rook + w_minor) <= 2 and (b_rook + b_minor) <= 2:
            return 2

    # One or both sides have queen(s): endgame only if the queen is
    # the sole non-pawn piece for that side
    else:
        w_others = w_rook + w_minor  # non-queen, non-pawn, non-king
        b_others = b_rook + b_minor
        if w_others == 0 and b_others == 0:
            return 2

    # --- Middlegame: everything else ---
    return 1


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

    bbs = _piece_bitboards(board)
    p12 = _bb_to_planes_batch(bbs)
    for i in range(12):
        if bbs[i]:
            planes[i] = p12[i]

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

    planes[16, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[17, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[18, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[19, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[20, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    if board.ep_square is not None:
        planes[21, 7 - board.ep_square // 8, board.ep_square % 8] = 1.0
    planes[22, :, :] = min(board.halfmove_clock, 100) / 100.0

    return planes


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — LC0 Engine Interface
# ═══════════════════════════════════════════════════════════════════════════

class LC0Engine:
    """Wraps LC0 via chess.engine.SimpleEngine for all-legal-moves evaluation."""

    def __init__(self, engine_path: str, weights_path: Optional[str] = None,
                 backend: str = "cuda-fp16", batch_size: int = 256,
                 threads: int = 1):
        self._engine = chess.engine.SimpleEngine.popen_uci(
            engine_path, timeout=60
        )
        config = {
            "UCI_ShowWDL": True,
            "PerPVCounters": True,
            "SmartPruningFactor": 0,
            "FpuStrategy": "absolute",
            "FpuValue": 0,
            "CPuct": 5.0,
            "PolicyTemperature": 10.0,
        }
        if weights_path:
            config["WeightsFile"] = weights_path
        config["Backend"] = backend
        config["MinibatchSize"] = batch_size
        config["Threads"] = threads
        self._engine.configure(config)

    def evaluate(self, board: chess.Board) -> Dict[str, Dict]:
        """Evaluate all legal moves. Returns {uci_move: {eval, wdl_w, wdl_d, wdl_l, nodes, depth}}"""
        legal_moves = list(board.legal_moves)
        n_legal = len(legal_moves)
        if n_legal == 0:
            return {}

        nodes = 5 * n_legal
        if nodes > 300:
            nodes = min(3 * n_legal, 500)
        nodes = max(nodes, 150)

        infos = self._engine.analyse(
            board, chess.engine.Limit(nodes=nodes),
            multipv=n_legal, info=chess.engine.INFO_ALL,
        )
        if not isinstance(infos, list):
            infos = [infos]

        results = {}
        for info in infos:
            pv = info.get("pv")
            if not pv:
                continue
            uci = pv[0].uci()
            if uci in results:
                continue

            score = info.get("score")
            cp = 0
            if score:
                pov = score.white()
                if pov.is_mate():
                    cp = 10000 if pov.mate() > 0 else -10000
                else:
                    cp = pov.score() or 0

            wdl_pov = info.get("wdl")
            if wdl_pov is not None:
                wdl = wdl_pov.white()
                wdl_w = wdl.wins / 1000.0
                wdl_d = wdl.draws / 1000.0
                wdl_l = wdl.losses / 1000.0
            else:
                wdl_w, wdl_d, wdl_l = 0.33, 0.34, 0.33

            results[uci] = {
                'eval': cp,
                'wdl_w': wdl_w, 'wdl_d': wdl_d, 'wdl_l': wdl_l,
                'nodes': info.get("nodes", 0),
                'depth': info.get("depth", 0),
                'policy_prob': 0.0,
            }

        return results

    def close(self):
        if self._engine:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Feature Construction
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
                   max_possible: int = 218,
                   model_max_possible: int = 220) -> Dict[str, torch.Tensor]:
    """Build all MIMO input tensors for a single position."""
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

    move_eval_list = []
    for m in legal_moves:
        uci = m.uci()
        ev = evals.get(uci, {})
        raw_eval = ev.get('eval', 0)
        stm_eval = raw_eval if color == 'White' else -raw_eval
        move_eval_list.append((m, uci, ev, stm_eval))
    move_eval_list.sort(key=lambda x: x[3], reverse=True)
    move_eval_list = move_eval_list[:max_possible]
    num_possible = len(move_eval_list)

    evals_stm = [x[3] for x in move_eval_list]
    best_eval_stm = max(evals_stm) if evals_stm else 0.0
    worst_eval_stm = min(evals_stm) if evals_stm else 0.0
    eval_range = best_eval_stm - worst_eval_stm

    current_planes = board_to_planes_fast(board, history)

    from_sqs = np.zeros(max_possible, dtype=np.int64)
    to_sqs   = np.zeros(max_possible, dtype=np.int64)
    promos   = np.zeros(max_possible, dtype=np.int64)
    for i, (move, uci, ev, stm_eval) in enumerate(move_eval_list):
        from_sqs[i] = move.from_square
        to_sqs[i]   = move.to_square
        promos[i]    = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                        chess.ROOK: 3, chess.QUEEN: 4}.get(move.promotion, 0)

    possible_scalars = np.zeros((max_possible, 13), dtype=np.float32)
    possible_mask = np.zeros(max_possible, dtype=np.float32)
    possible_mask[:num_possible] = 1.0

    move_ucis = []
    for i, (move, uci, ev, stm_eval) in enumerate(move_eval_list):
        move_ucis.append(uci)
        wdl_w = ev.get('wdl_w', 0.0033) * 100.0
        wdl_d = ev.get('wdl_d', 0.0034) * 100.0
        wdl_l = ev.get('wdl_l', 0.0033) * 100.0

        nodes_raw = ev.get('nodes', 1)
        depth = ev.get('depth', 1)
        move_quality = (stm_eval - worst_eval_stm) / eval_range if eval_range > 0 else 1.0

        piece = board.piece_at(move.from_square)
        piece_sym = piece.symbol().upper() if piece else 'P'
        piece_val = PIECE_MAP.get(piece_sym, 1/6)

        is_capture = 0.0
        target = board.piece_at(move.to_square)
        if target is not None:
            is_capture = 1.0
        if board.ep_square == move.to_square and piece_sym == 'P':
            is_capture = 1.0

        board.push(move)
        is_check = 1.0 if board.is_check() else 0.0
        is_checkmate = 1.0 if board.is_checkmate() else 0.0
        board.pop()

        promotion = 1.0 if move.promotion else 0.0

        possible_scalars[i] = [
            stm_eval / 1000.0, wdl_w, wdl_d, wdl_l,
            math.log1p(nodes_raw) / 20.0, depth / 40.0,
            move_quality, piece_val, is_capture, is_check, is_checkmate,
            0.0,  # is_mistake_move — filled after EV computation
            0.0,  # is_excellent_move — filled after EV computation
        ]

    # --- Per-move mistake/excellent flags (EV from STM perspective) ---
    is_white = (color == 'White')
    if num_possible > 0:
        w_pcts = possible_scalars[:num_possible, 1] / 100.0
        d_pcts = possible_scalars[:num_possible, 2] / 100.0
        l_pcts = possible_scalars[:num_possible, 3] / 100.0
        if is_white:
            stm_evs = w_pcts + 0.5 * d_pcts
        else:
            stm_evs = l_pcts + 0.5 * d_pcts
        best_ev_idx = int(np.argmax(stm_evs))
        best_ev_val = stm_evs[best_ev_idx]
        drops = best_ev_val - stm_evs
        for j in range(num_possible):
            possible_scalars[j, 11] = 1.0 if drops[j] > 0.25 else 0.0
            possible_scalars[j, 12] = 1.0 if (drops[j] < 0.025 or j == best_ev_idx) else 0.0
        frac_mistake_moves = float(possible_scalars[:num_possible, 11].sum()) / num_possible
        frac_excellent_moves = float(possible_scalars[:num_possible, 12].sum()) / num_possible
    else:
        frac_mistake_moves = 0.0
        frac_excellent_moves = 0.0

    eval_stm = best_eval_stm
    if move_eval_list:
        best_ev = move_eval_list[0][2]
        white_win_before = best_ev.get('wdl_w', 0.0033) * 100.0
        white_draw_before = best_ev.get('wdl_d', 0.0034) * 100.0
        white_loss_before = best_ev.get('wdl_l', 0.0033) * 100.0
    else:
        white_win_before, white_draw_before, white_loss_before = 0.33, 0.34, 0.33

    in_check = 1.0 if board.is_check() else 0.0
    eval_std = float(np.std(evals_stm)) / 1000.0 if len(evals_stm) > 1 else 0.0
    captures_frac = sum(possible_scalars[i, 8] for i in range(num_possible)) / max(num_possible, 1)
    checks_frac = sum(possible_scalars[i, 9] for i in range(num_possible)) / max(num_possible, 1)
    num_candidates = num_possible / model_max_possible

    tabular = np.array([
        clock_time / 3600.0, w_elo / 3000.0, b_elo / 3000.0,
        (w_elo - b_elo) / 1000.0, min(move_no, 200) / 200.0,
        1.0 if color == 'White' else 0.0, eval_stm / 1000.0,
        white_win_before, white_draw_before, white_loss_before,
        initial_time / 3600.0, increment / 60.0, prev_capture, in_check,
        eval_std, captures_frac, checks_frac, num_candidates,
        frac_mistake_moves, frac_excellent_moves,
    ], dtype=np.float32)

    # Phase anchor (opening=ply, endgame=material, else=middlegame)
    ply = (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1)
    game_phase = compute_game_phase_from_board(board, ply)

    return {
        'current_planes':   torch.from_numpy(current_planes).unsqueeze(0),
        'possible_from_sq': torch.from_numpy(from_sqs).unsqueeze(0),
        'possible_to_sq':   torch.from_numpy(to_sqs).unsqueeze(0),
        'possible_promo':   torch.from_numpy(promos).unsqueeze(0),
        'possible_scalars': torch.from_numpy(possible_scalars).unsqueeze(0),
        'possible_mask':    torch.from_numpy(possible_mask).unsqueeze(0),
        'tabular':          torch.from_numpy(tabular).unsqueeze(0),
        'game_phase':       game_phase,
        '_move_ucis':      move_ucis,
        '_n_legal':        n_legal,
        '_color':          color,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — MIMO V5 Predictor
# ═══════════════════════════════════════════════════════════════════════════

class MIMOPredictor:
    """Load a trained MIMO V5 checkpoint and run inference."""

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device)
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get('config', {})
        eff = dict(cfg.get('effective_config', {}) or {})
        comp_ver = cfg.get('component_versions', {})

        def _arch(default, *keys):
            """Resolve from effective_config → top-level config → default."""
            for src in (eff, cfg):
                for k in keys:
                    v = src.get(k)
                    if v is not None:
                        return v
            return default

        if eff:
            print(f"[V5] Resolving architecture from checkpoint effective_config")
        else:
            print(f"[V5] WARNING: No effective_config — using hardcoded defaults")

        self.model = ChessMIMOModelV5(
            cnn_channels=_arch(128, 'cnn_channels'),
            num_res_blocks=_arch(6, 'num_res_blocks', 'res_blocks'),
            tabular_dim=_arch(20, 'tabular_dim'),
            max_possible=_arch(220, 'max_possible'),
            hidden_dim=_arch(256, 'hidden_dim'),
            move_scalar_dim=_arch(13, 'move_scalar_dim'),
            sq_embed_dim=_arch(48, 'sq_embed_dim'),
            expert_hidden=_arch(160, 'expert_hidden'),
            mistake_expert_ver=comp_ver.get('mistake_expert', 'default'),
            time_expert_ver=comp_ver.get('time_expert', 'default'),
            wdl_expert_ver=comp_ver.get('wdl_expert', 'default'),
            move_head_ver=comp_ver.get('move_head', 'default'),
            contrastive_embed_dim=_arch(64, 'contrastive_embed_dim'),
            contrastive_hidden_dim=_arch(128, 'contrastive_hidden_dim'),
            contrastive_margin=_arch(1.0, 'contrastive_margin'),
            use_phase_experts=_arch(True, 'use_phase_experts'),
            phase_hidden_dim=_arch(64, 'phase_hidden_dim'),
            num_phases=_arch(3, 'num_phases'),
            use_film=_arch(True, 'use_film'),
            film_hidden_dim=_arch(64, 'film_hidden_dim'),
            use_tactical_enrichment=_arch(False, 'use_tactical_enrichment'),
            tactical_preprocessor_config=cfg.get('tactical_preprocessor_config'),
        ).to(self.device)

        state_dict = ckpt['model_state_dict']
        # Strip _orig_mod. prefix added by torch.compile() wrapping
        state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[V5] {len(missing)} missing keys (new V5 modules with fresh weights)")
        self.model.eval()
        self.max_possible = _arch(220, 'max_possible')
        n_params = sum(p.numel() for p in self.model.parameters())
        epoch = ckpt.get('epoch', '?')
        print(f"Loaded epoch {epoch} — {n_params:,} params")
        print(f"V5 features: contrastive_dim={self.model.contrastive_embed_dim}, "
              f"phase_experts={self.model.use_phase_experts}, "
              f"move_head={comp_ver.get('move_head', 'default')}")

    @torch.no_grad()
    def predict(self, features: Dict[str, torch.Tensor]) -> Dict:
        """Run forward pass, return decoded predictions."""
        cp = features['current_planes'].to(self.device)
        pf = features['possible_from_sq'].to(self.device)
        pt = features['possible_to_sq'].to(self.device)
        pp = features['possible_promo'].to(self.device)
        ps = features['possible_scalars'].to(self.device)
        pm = features['possible_mask'].to(self.device)
        tab = features['tabular'].to(self.device)

        with autocast('cuda', enabled=(self.device.type == 'cuda')):
            out = self.model(cp, pf, pt, pp, ps, pm, tab, actual_idx=None)

        move_probs = torch.softmax(out['move_logits'], dim=-1).cpu().numpy()[0]
        mistake_prob = out['mistake_prob'].sigmoid().cpu().numpy()[0].item()
        wdl_before = out.get('win_prob_before')
        time_log = out['time_spent'].cpu().numpy()[0].item()

        preds = {
            'move_probs': move_probs,
            'mistake_prob': mistake_prob,
            'predicted_time_s': float(np.expm1(time_log)),
        }
        if wdl_before is not None:
            preds['wdl_before'] = wdl_before.cpu().numpy()[0]

        # Phase weights (V5)
        if 'phase_weights' in out:
            preds['phase_weights'] = out['phase_weights'].cpu().numpy()[0]

        # Game phase (from features, always present)
        if 'game_phase' in features:
            preds['game_phase'] = features['game_phase']

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
    print(f"  MIMO V5 Prediction — {color} to move")
    print(f"  FEN: {board.fen()}")
    if 'game_phase' in preds:
        phase_names = ['Opening', 'Middlegame', 'Endgame']
        print(f"  Game phase: {phase_names[preds['game_phase']]}")
    print(f"  Legal moves: {n_legal}")
    print(f"{'═' * 60}")

    # --- WDL Before ---
    if 'wdl_before' in preds:
        wdl = preds['wdl_before']
        print(f"\n  Game outcome prediction (WDL from White's view):")
        print(f"    Win {wdl[0]:.1%}  |  Draw {wdl[1]:.1%}  |  Loss {wdl[2]:.1%}")

    # --- Phase weights (V5) ---
    if 'phase_weights' in preds:
        pw = preds['phase_weights']
        phase_names = ['Opening', 'Middlegame', 'Endgame']
        phase_str = '  |  '.join(f"{n} {pw[i]:.1%}" for i, n in enumerate(phase_names[:len(pw)]))
        print(f"\n  Game phase: {phase_str}")

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
        description='MIMO V5 real-time inference — predict human chess behavior')

    # Position
    parser.add_argument('--fen', required=True, help='FEN of the position to analyze')
    parser.add_argument('--history', type=str, default=None,
                        help='Comma-separated UCI moves from game start (e.g. e2e4,e7e5,g1f3)')

    # Game context
    parser.add_argument('--white-elo', type=int, default=1500)
    parser.add_argument('--black-elo', type=int, default=1500)
    parser.add_argument('--time-control', type=str, default='300+3')
    parser.add_argument('--clock-time', type=float, default=None)

    # Engine
    parser.add_argument('--lc0', required=True, help='Path to LC0 executable')
    parser.add_argument('--lc0-weights', type=str, default=None)

    # Model
    parser.add_argument('--checkpoint', required=True, help='Path to MIMO V5 checkpoint (.pt)')
    parser.add_argument('--max-possible', type=int, default=218)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # Output
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--json', action='store_true', help='Output results as JSON')

    args = parser.parse_args()

    # --- Set up board ---
    board = chess.Board(args.fen)
    color = 'White' if board.turn == chess.WHITE else 'Black'
    move_no = board.fullmove_number

    # --- Parse move history ---
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
        history = history[-2:] if history else None

    # --- Detect prev_capture ---
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
        elapsed_est = move_no * 5
        clock_time = max(0, initial_time - elapsed_est + increment * move_no)

    game_meta = {
        'white_elo': args.white_elo,
        'black_elo': args.black_elo,
        'time_control': args.time_control,
        'clock_time': clock_time,
        'move_no': move_no,
        'prev_capture': prev_capture,
    }

    # --- Load MIMO V5 model ---
    predictor = MIMOPredictor(args.checkpoint, args.device)

    # --- LC0 evaluation ---
    print(f"Starting LC0 ({args.lc0}) ...")
    engine = LC0Engine(args.lc0, args.lc0_weights)
    t0 = time.time()
    evals = engine.evaluate(board)
    lc0_ms = (time.time() - t0) * 1000
    engine.close()
    print(f"LC0 evaluated {len(evals)} moves in {lc0_ms:.0f}ms")

    if not evals:
        print("ERROR: LC0 returned no evaluations.")
        sys.exit(1)

    # --- Feature construction ---
    t0 = time.time()
    features = build_features(board, history, evals, game_meta, args.max_possible,
                              model_max_possible=predictor.max_possible)
    feat_ms = (time.time() - t0) * 1000
    print(f"Features built in {feat_ms:.1f}ms")

    # --- MIMO inference ---
    t0 = time.time()
    preds = predictor.predict(features)
    infer_ms = (time.time() - t0) * 1000
    print(f"MIMO forward pass in {infer_ms:.1f}ms")

    # --- Output ---
    results = format_results(features, preds, board, args.top_k)

    if args.json:
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
