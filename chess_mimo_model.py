#!/usr/bin/env python3
"""
chess_mimo_model.py — Multi-Input Multi-Output Chess Behavior Prediction Model

Architecture overview:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         INPUTS                                      │
    │  current_planes (B,23,8,8)  tabular (B,10)  possible_planes (B,M,47,8,8) │
    │                                             possible_scalars (B,M,6)│
    └────────┬──────────────────────┬─────────────────┬───────────────────┘
             │                      │                 │
     ┌───────▼───────┐   ┌─────────▼──────┐  ┌───────▼────────┐
     │  Shared CNN    │   │ Tabular MLP    │  │  Shared CNN    │
     │  6 ResBlocks   │   │ 3-layer+LN     │  │  (same weights)│
     │  + SE blocks   │   │ → 64-dim       │  │  + scalar MLP  │
     │  → 128-dim     │   └────────┬───────┘  │  → 256-dim/move│
     └───────┬────────┘            │          └───────┬────────┘
             │                     │                  │
             └──────────┬──────────┘          ┌───────▼────────┐
                        │                     │  Cross-Attention│
               ┌────────▼────────┐            │  Q=fused_ctx   │
               │  Concat + MLP   │◄───────────│  K,V=move_embs │
               │  → 256 global   │            └───────┬────────┘
               └────────┬────────┘                    │
                        │                             │
        ┌───────┬───────┼───────┬───────┐             │
        ▼       ▼       ▼       ▼       ▼             │
    ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐         │
    │move  ││mistak││WDL   ││WDL   ││time  │         │
    │logits││e_prob││before││after ││spent │         │
    │(per- ││      ││(mask)││(+act)││      │         │
    │move) ││      ││      ││      ││      │         │
    └──────┘└──────┘└──────┘└──────┘└──────┘

Masking strategy:
    - move_logits: sees all possible moves, picks which one human played
    - mistake_prob: sees position + candidates, NOT which was played or game result
    - win_prob_before: actual move MASKED from cross-attention, no game result
    - win_prob_after: sees everything incl. actual move embedding, no game result
    - time_spent: time_spent excluded from tabular features entirely

Author: Sskeer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """Channel-attention (SE) block: learn per-channel importance weights."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


class ResBlock(nn.Module):
    """Pre-activation residual block (BN → GELU → Conv)."""

    def __init__(self, channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.bn1(x))
        out = self.conv1(out)
        out = F.gelu(self.bn2(out))
        out = self.conv2(out)
        return out + residual


class BoardEncoder(nn.Module):
    """
    6-residual-block CNN with SE attention for 23-plane chess positions.

    Accepts either (B, 23, 8, 8) or flattened batches from possible moves
    reshaped to (B*M, 23, 8, 8) externally.
    """

    def __init__(self, in_planes: int = 23, channels: int = 128, num_res_blocks: int = 6):
        super().__init__()
        # Stem: project 23 input planes to `channels`
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        # Residual tower with SE every 2 blocks
        layers = []
        for i in range(num_res_blocks):
            layers.append(ResBlock(channels))
            if (i + 1) % 2 == 0:
                layers.append(SqueezeExcitation(channels))
        self.tower = nn.Sequential(*layers)
        # Global average pool
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 23, 8, 8) → (N, channels)"""
        x = self.stem(x)
        x = self.tower(x)
        x = self.pool(x).flatten(1)
        return x


class TabularEncoder(nn.Module):
    """
    3-layer MLP with LayerNorm for scalar features.

    10 input features:
      time_remaining/3600, white_elo/3000, black_elo/3000, elo_diff/1000,
      move_no/200, color (0|1), eval_stm_before/1000,
      stm_win_before, draw_perc_before, stm_loss_before

    All evals and WDL normalised to side-to-move (STM) perspective.
    NO time_spent — that's a prediction target.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )
        self.out_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PossibleMoveScalarEncoder(nn.Module):
    """Small MLP to encode per-move scalar features.

    6 features (all in STM perspective):
      eval_stm/1000, stm_win_perc, draw_perc, stm_loss_perc,
      log1p(nodes)/20, depth/40
    """

    def __init__(self, input_dim: int = 6, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.GELU(),
            nn.Linear(48, output_dim),
            nn.GELU(),
        )
        self.out_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, M, input_dim) → (B, M, output_dim)  OR  (N, input_dim) → (N, output_dim)"""
        if x.ndim == 2:
            return self.net(x)
        B, M, D = x.shape
        return self.net(x.reshape(B * M, D)).reshape(B, M, -1)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class ChessMIMOModel(nn.Module):
    """
    Multi-Input Multi-Output model for predicting human chess behaviour.

    Parameters
    ----------
    cnn_channels : int
        Width of the residual CNN (default 128).
    num_res_blocks : int
        Number of residual blocks in the CNN (default 6).
    tabular_dim : int
        Number of scalar input features (default 10).
    max_possible : int
        Maximum number of candidate moves per position (default 40).
    hidden_dim : int
        Dimension of the global fused representation (default 256).
    num_attn_heads : int
        Heads in the cross-attention over candidate moves (default 4).
    dropout : float
        Dropout rate in fusion / heads (default 0.2).
    move_scalar_dim : int
        Number of scalar features per candidate move (default 6).
    """

    def __init__(
        self,
        cnn_channels: int = 128,
        num_res_blocks: int = 6,
        tabular_dim: int = 18,
        max_possible: int = 220,
        hidden_dim: int = 256,
        num_attn_heads: int = 4,
        dropout: float = 0.2,
        move_scalar_dim: int = 12,
    ):
        super().__init__()
        self.max_possible = max_possible
        self.hidden_dim = hidden_dim

        # ---- Encoders (shared CNN for current + possible boards) ----
        self.board_encoder = BoardEncoder(
            in_planes=23, channels=cnn_channels, num_res_blocks=num_res_blocks
        )
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_dim, output_dim=64, dropout=dropout * 0.5
        )
        self.move_scalar_encoder = PossibleMoveScalarEncoder(
            input_dim=move_scalar_dim, output_dim=32
        )

        # ---- Possible-move projection ----
        # CNN output (128) + scalar encoder output (32) → hidden_dim
        move_emb_dim = cnn_channels + 32
        self.move_proj = nn.Sequential(
            nn.Linear(move_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ---- Cross-attention: context queries, candidate-move keys/values ----
        # Query dimension = cnn_channels + tabular_out = 128 + 64 = 192
        self.query_proj = nn.Linear(cnn_channels + 64, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_attn_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # ---- Gated cross-attention ----
        # Learned gate controls how much attention output contributes to fusion.
        # Worst case: gate ≈ 1.0 everywhere → equivalent to ungated architecture.
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # ---- Fusion MLP ----
        # Input: board_emb (128) + tabular (64) + attn_output (256) = 448
        fusion_in = cnn_channels + 64 + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ---- Output Heads ----

        # 1. Move logits: per-move score from move embedding + global context + aux feedback
        #    Input: global_hidden (256) + move_emb (256) + aux_feedback (5) = 517
        self.move_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 2. Mistake probability (binary — outputs logits, sigmoid applied in loss/inference)
        self.mistake_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 3. Win prob before (WDL, 3-way) — uses masked context
        self.wdl_before_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

        # 4. Win prob after (WDL, 3-way) — sees actual move
        #    Input: global_hidden (256) + actual_move_emb (256) = 512
        self.wdl_after_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

        # 5. Time spent (log-scale scalar)
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Kaiming init for conv/linear, zeros for final head biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Core encoding helpers
    # ------------------------------------------------------------------

    def _encode_possible_moves(
        self,
        possible_planes: torch.Tensor,
        possible_scalars: torch.Tensor,
        possible_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode each candidate move into an embedding vector.

        Uses static reshape instead of boolean indexing so the entire
        CNN + projection is compilable by torch.compile (no graph breaks).
        Padding moves (zero planes) are processed through the CNN but
        masked to zero afterward.

        Parameters
        ----------
        possible_planes : (B, M, 23, 8, 8)
        possible_scalars : (B, M, 12)
        possible_mask : (B, M)  — 1.0 for valid moves, 0.0 for padding

        Returns
        -------
        move_emb : (B, M, hidden_dim)
        """
        B, M, C, H, W = possible_planes.shape

        # Static reshape — fully compilable, no graph breaks
        flat_planes = possible_planes.reshape(B * M, C, H, W)
        flat_scalars = possible_scalars.reshape(B * M, -1)

        # CNN + scalar encoder on all moves (padding included)
        flat_cnn = self.board_encoder(flat_planes)               # (B*M, cnn_channels)
        scalar_out = self.move_scalar_encoder(flat_scalars)      # (B*M, 32)
        combined = torch.cat([flat_cnn, scalar_out], dim=-1)     # (B*M, cnn_channels+32)
        proj = self.move_proj(combined)                          # (B*M, hidden_dim)

        # Reshape back and zero out padding embeddings
        move_emb = proj.reshape(B, M, self.hidden_dim)
        move_emb = move_emb * possible_mask.unsqueeze(-1)        # mask padding
        return move_emb

    def _cross_attend(
        self,
        context: torch.Tensor,
        move_emb: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-attention: context attends to candidate-move embeddings.

        Parameters
        ----------
        context : (B, ctx_dim)   — board_emb ∥ tabular_emb
        move_emb : (B, M, hidden_dim)
        key_padding_mask : (B, M)  — True where padded (invalid)

        Returns
        -------
        attn_out : (B, hidden_dim)
        """
        query = self.query_proj(context).unsqueeze(1)        # (B, 1, hidden_dim)
        attn_out, _ = self.cross_attn(
            query, move_emb, move_emb, key_padding_mask=key_padding_mask
        )
        attn_out = self.attn_norm(attn_out.squeeze(1))       # (B, hidden_dim)
        return attn_out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        current_planes: torch.Tensor,
        possible_planes: torch.Tensor,
        possible_scalars: torch.Tensor,
        possible_mask: torch.Tensor,
        tabular: torch.Tensor,
        actual_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with built-in masking for win_prob_before.

        Parameters
        ----------
        current_planes : (B, 23, 8, 8)
        possible_planes : (B, M, 23, 8, 8)
        possible_scalars : (B, M, 11)
        possible_mask : (B, M)   — 1.0 for valid moves, 0.0 for padding
        tabular : (B, 18)
        actual_idx : (B,) long or None
            Index of the move actually played within the possible-moves list.
            Required during training for:
              - win_prob_before masking
              - win_prob_after (selects actual move embedding)
            At inference time, pass None to get everything except those two heads.

        Returns
        -------
        dict with keys:
            move_logits     (B, M)
            mistake_prob    (B, 1)
            win_prob_before (B, 3)   — only if actual_idx provided
            win_prob_after  (B, 3)   — only if actual_idx provided
            time_spent      (B, 1)
        """
        B = current_planes.shape[0]
        M = possible_planes.shape[1]

        # --- Clamp scalar inputs to float16-safe range ---
        # Prevents inf/NaN from sentinel values (e.g. INT_MAX evals, extreme times)
        tabular = tabular.clamp(-1e4, 1e4)
        possible_scalars = possible_scalars.clamp(-1e4, 1e4)

        # --- Encode current board ---
        board_emb = self.board_encoder(current_planes)       # (B, 128)

        # --- Encode tabular ---
        tab_emb = self.tabular_encoder(tabular)              # (B, 64)

        # --- Encode all candidate moves (valid only — skips masked padding) ---
        move_emb = self._encode_possible_moves(possible_planes, possible_scalars, possible_mask)
        # move_emb: (B, M, hidden_dim)

        # --- Context vector ---
        context = torch.cat([board_emb, tab_emb], dim=-1)    # (B, 192)

        # Padding mask for attention (True = ignore)
        pad_mask = (possible_mask == 0)                      # (B, M)

        # --- Full cross-attention (all moves visible) ---
        attn_out = self._cross_attend(context, move_emb, pad_mask)  # (B, hidden_dim)
        attn_out = self.attn_gate(attn_out) * attn_out              # gated

        # --- Fusion ---
        fused = torch.cat([board_emb, tab_emb, attn_out], dim=-1)  # (B, 448)
        global_hidden = self.fusion(fused)                          # (B, 256)

        outputs: Dict[str, torch.Tensor] = {}

        # ============================================================
        # AUXILIARY HEADS (computed first for feedback into move head)
        # ============================================================

        # HEAD 2: Mistake probability
        outputs['mistake_prob'] = self.mistake_head(global_hidden)   # (B, 1)

        # HEAD 5: Time spent (log-scale)
        outputs['time_spent'] = self.time_head(global_hidden)        # (B, 1)

        # HEAD 3: Win probability before move
        # Uses the same global_hidden as other heads — no masking needed.
        # The cross-attention sees all legal moves as anonymous K/V pairs;
        # which move was actually played is not encoded in the attention.
        wdl_before = F.softmax(
            self.wdl_before_head(global_hidden), dim=-1
        )  # (B, 3)
        outputs['win_prob_before'] = wdl_before

        # ============================================================
        # HEAD 1: Move logits — with auxiliary feedback (packed for efficiency)
        # ============================================================
        # Detach auxiliary predictions so move loss doesn't corrupt aux heads
        aux_feedback = torch.cat([
            outputs['mistake_prob'].detach().sigmoid(),   # (B, 1) — logits → prob for aux
            outputs['time_spent'].detach(),     # (B, 1)
            wdl_before.detach(),                # (B, 3)
        ], dim=-1)  # (B, 5)

        # Pack valid moves only — skip ~86% of move_head compute
        valid = ~pad_mask  # (B, M) — True for valid moves
        # Use torch.where for valid indices — always runs (no data-dependent branch)
        batch_idx, move_idx = torch.where(valid)  # (N_valid,)
        flat_move_emb = move_emb[batch_idx, move_idx]  # (N_valid, hidden_dim)
        flat_global = global_hidden[batch_idx]  # (N_valid, 256)
        flat_aux = aux_feedback[batch_idx]  # (N_valid, 5)
        
        # Compute scores for valid moves only
        flat_input = torch.cat([flat_global, flat_move_emb, flat_aux], dim=-1)  # (N_valid, 517)
        flat_scores = self.move_head(flat_input).squeeze(-1)  # (N_valid,)
        
        # Scatter back to (B, M) with -inf for padding (ensure dtype match for AMP)
        move_scores = torch.full((B, M), float('-inf'), device=move_emb.device, dtype=torch.float32)
        move_scores[batch_idx, move_idx] = flat_scores.to(move_scores.dtype)
        
        outputs['move_logits'] = move_scores

        # ============================================================
        # HEAD 4: Win probability after move (sees actual move)
        # ============================================================
        if actual_idx is not None:
            safe_aidx = actual_idx.clamp(min=0)
            idx_exp = safe_aidx.unsqueeze(-1).expand(-1, self.hidden_dim)  # (B, 256)
            actual_move_emb = move_emb.gather(1, idx_exp.unsqueeze(1)).squeeze(1)
            wdl_after_input = torch.cat([global_hidden, actual_move_emb], dim=-1)
            outputs['win_prob_after'] = F.softmax(
                self.wdl_after_head(wdl_after_input), dim=-1
            )  # (B, 3)

        return outputs


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class MIMOLoss(nn.Module):
    """
    Multi-task loss with learnable uncertainty-based weighting
    (Kendall, Gal & Cipolla 2018).

    Each task has a learnable log-variance σ² parameter.
    total_loss = Σ (1/(2σ²_i)) * loss_i + log(σ²_i)
    """

    HEADS = ['move_logits', 'mistake_prob', 'win_prob_before', 'win_prob_after', 'time_spent']

    def __init__(self):
        super().__init__()
        # Learnable log-variance for each task (initialised to 0 → σ²=1 → weight=0.5)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in self.HEADS
        })

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Targets dict expected keys:
            move_idx        (B,) long
            is_mistake      (B,) float 0/1
            win_prob_before (B, 3) float — one-hot WDL
            win_prob_after  (B, 3) float — one-hot WDL
            time_spent_log  (B,) float — log1p(seconds)
        """
        losses: Dict[str, torch.Tensor] = {}

        # 1. Move prediction — cross-entropy (ignore actual_idx=-1 = move not found)
        if 'move_logits' in predictions and 'move_idx' in targets:
            losses['move_logits'] = F.cross_entropy(
                predictions['move_logits'], targets['move_idx'],
                ignore_index=-1,
            )

        # 2. Mistake — binary cross-entropy (with logits for AMP safety)
        #    Mask out examples where actual_idx == -1 (no valid move found)
        if 'mistake_prob' in predictions and 'is_mistake' in targets:
            valid_mask = targets['move_idx'] >= 0
            # Always compute BCE on full batch; masked positions contribute 0 via multiplication
            raw_bce = F.binary_cross_entropy_with_logits(
                predictions['mistake_prob'].squeeze(-1),
                targets['is_mistake'],
                reduction='none',
            )
            # Mask invalid positions and mean over valid ones
            masked_bce = raw_bce * valid_mask.float()
            n_valid = valid_mask.float().sum().clamp(min=1.0)
            losses['mistake_prob'] = masked_bce.sum() / n_valid

        # 3. WDL before — stable cross-entropy against target distribution
        #    (replaces F.kl_div which produces NaN with one-hot targets:
        #     0 * log(0) = 0 * -inf = NaN in IEEE 754)
        if 'win_prob_before' in predictions and 'win_prob_before' in targets:
            log_pred = torch.log(predictions['win_prob_before'].clamp(min=1e-8))
            losses['win_prob_before'] = -(targets['win_prob_before'] * log_pred).sum(dim=-1).mean()

        # 4. WDL after — same stable formulation
        if 'win_prob_after' in predictions and 'win_prob_after' in targets:
            log_pred = torch.log(predictions['win_prob_after'].clamp(min=1e-8))
            losses['win_prob_after'] = -(targets['win_prob_after'] * log_pred).sum(dim=-1).mean()

        # 5. Time spent — Huber loss (robust to outliers)
        if 'time_spent' in predictions and 'time_spent_log' in targets:
            losses['time_spent'] = F.huber_loss(
                predictions['time_spent'].squeeze(-1),
                targets['time_spent_log'],
                delta=2.0,
            )

        # Uncertainty-weighted combination
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for name, loss_val in losses.items():
            log_var = self.log_vars[name]
            precision = torch.exp(-log_var)
            total = total + precision * loss_val + log_var

        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict['total'] = total.item()
        # Also log effective weights
        for name in losses:
            loss_dict[f'w_{name}'] = torch.exp(-self.log_vars[name]).item()

        return total, loss_dict


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(42)
    B, M = 4, 40

    model = ChessMIMOModel(max_possible=M)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Dummy inputs
    current = torch.randn(B, 23, 8, 8)
    poss_planes = torch.randn(B, M, 23, 8, 8)
    poss_scalars = torch.randn(B, M, 6)
    poss_mask = torch.ones(B, M)
    poss_mask[:, 25:] = 0  # last 15 are padding
    tabular = torch.randn(B, 10)
    actual_idx = torch.randint(0, 25, (B,))

    outputs = model(current, poss_planes, poss_scalars, poss_mask, tabular, actual_idx)
    print("\nOutputs:")
    for k, v in outputs.items():
        print(f"  {k:20s} {tuple(v.shape)}")

    # Loss
    targets = {
        'move_idx': torch.randint(0, 25, (B,)),
        'is_mistake': torch.randint(0, 2, (B,)).float(),
        'win_prob_before': F.softmax(torch.randn(B, 3), dim=-1),
        'win_prob_after': F.softmax(torch.randn(B, 3), dim=-1),
        'time_spent_log': torch.rand(B) * 4,
    }
    criterion = MIMOLoss()
    loss, ld = criterion(outputs, targets)
    print(f"\nTotal loss: {loss.item():.4f}")
    for k, v in ld.items():
        print(f"  {k:25s} {v:.4f}")
