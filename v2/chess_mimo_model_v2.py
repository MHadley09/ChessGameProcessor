#!/usr/bin/env python3
"""
chess_mimo_model_v2.py — Specialist-Expert Multi-Input Multi-Output Chess Model

V2 architecture adds specialist ExpertModules for each auxiliary head
and feeds their hidden states (detached) into a deeper move head via
cross-feed fusion.

Architecture overview:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         INPUTS                                      │
    │  current_planes (B,23,8,8)  tabular (B,18)  possible_planes (B,M,47,8,8) │
    │                                             possible_scalars (B,M,12)│
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
               │  Trunk (1-layer)│◄───────────│  K,V=move_embs │
               │  → 256 global   │            └────────────────┘
               └────────┬────────┘
                        │
         ┌──────────────┼──────────────────────────┐
         │              │                          │
    ┌────▼────┐   ┌─────▼─────┐   ┌──────┐   ┌────▼────┐
    │WDL Before│   │ Mistake   │   │ Time │   │WDL After│
    │ Expert   │   │  Expert   │   │Expert│   │ Expert  │
    │(masked   │   │(trunk)    │   │(trunk│   │(trunk + │
    │ trunk)   │   │→h:128     │   │→h:128│   │ actual  │
    │→h:128    │   │→out:1     │   │→out:1│   │ move)   │
    │→out:3    │   └─────┬─────┘   └──┬───┘   │→h:128   │
    └────┬─────┘         │            │       │→out:3   │
         │               │            │       └─────────┘
         └───────┬───────┘            │
                 │ (detach)           │ (detach)
                 ▼                    │
    ┌───────────────────────────────────────────┐
    │  Cross-Feed Fusion                        │
    │  [trunk(256) + wdl_h(128) + mis_h(128)    │
    │   + time_h(128)] = 640                    │
    │  + move_emb(256) per move = 896           │
    │  → 3-layer MLP → per-move score           │
    └───────────────────────────────────────────┘

Masking strategy (same as V1):
    - wdl_before: actual move MASKED from cross-attention, no game result
    - wdl_after: sees everything incl. actual move embedding
    - mistake/time: sees position + all candidates via unmasked trunk
    - move_logits: cross-feed from expert hiddens + move embeddings

Author: Sskeer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Building Blocks (identical to V1)
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

    18 input features (expanded from V1's 10):
      time_remaining/3600, white_elo/3000, black_elo/3000, elo_diff/1000,
      move_no/200, color (0|1), eval_stm/1000,
      stm_win_before, draw_perc_before, stm_loss_before,
      initial_time/3600, increment/60, prev_capture, in_check,
      eval_std/1000, captures_frac, checks_frac, num_candidates

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

    12 features (all in STM perspective):
      eval_stm/1000, stm_win_perc, draw_perc, stm_loss_perc,
      log1p(nodes)/20, depth/40, move_quality, piece_val,
      is_capture, is_check, is_checkmate, policy_prob
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
# Expert Module (NEW in V2)
# ---------------------------------------------------------------------------

class ExpertModule(nn.Module):
    """
    Reusable specialist encoder for auxiliary heads.

    Returns both (hidden, output) — hidden is used for cross-feed into
    the move head, output is the task-specific prediction (pre-activation).

    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    hidden_dim : int
        Width of each hidden layer in the backbone.
    output_dim : int
        Dimension of the output (e.g. 3 for WDL, 1 for scalar).
    n_layers : int
        Number of hidden layers in the backbone (default 2).
    dropout : float
        Dropout rate (default 0.2).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, input_dim)

        Returns
        -------
        hidden : (B, hidden_dim)  — last backbone hidden state
        output : (B, output_dim)  — task prediction (pre-activation)
        """
        hidden = self.backbone(x)
        output = self.head(hidden)
        return hidden, output


# ---------------------------------------------------------------------------
# Main Model V2
# ---------------------------------------------------------------------------

class ChessMIMOModelV2(nn.Module):
    """
    Multi-Input Multi-Output model V2 with specialist expert heads
    and cross-feed fusion into a deeper move head.

    Parameters
    ----------
    cnn_channels : int
        Width of the residual CNN (default 128).
    num_res_blocks : int
        Number of residual blocks in the CNN (default 6).
    tabular_dim : int
        Number of scalar input features (default 18).
    max_possible : int
        Maximum number of candidate moves per position (default 40).
    hidden_dim : int
        Dimension of the global fused representation (default 256).
    num_attn_heads : int
        Heads in the cross-attention over candidate moves (default 4).
    dropout : float
        Dropout rate in fusion / heads (default 0.2).
    move_scalar_dim : int
        Number of scalar features per candidate move (default 12).
    expert_hidden : int
        Hidden dimension for all expert modules (default 128).
    expert_layers : int
        Number of hidden layers in each expert module (default 2).
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
        expert_hidden: int = 128,
        expert_layers: int = 2,
    ):
        super().__init__()
        self.max_possible = max_possible
        self.hidden_dim = hidden_dim
        self.expert_hidden = expert_hidden

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
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # ---- Trunk (single-layer fusion — lighter than V1's 2-layer) ----
        fusion_in = cnn_channels + 64 + hidden_dim  # 448
        self.trunk = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Expert Modules ----

        # WDL before: input = (masked) trunk output
        self.wdl_before_expert = ExpertModule(
            input_dim=hidden_dim, hidden_dim=expert_hidden,
            output_dim=3, n_layers=expert_layers, dropout=dropout,
        )

        # WDL after: input = trunk + actual_move_emb
        self.wdl_after_expert = ExpertModule(
            input_dim=hidden_dim * 2, hidden_dim=expert_hidden,
            output_dim=3, n_layers=expert_layers, dropout=dropout,
        )

        # Mistake: input = trunk output
        self.mistake_expert = ExpertModule(
            input_dim=hidden_dim, hidden_dim=expert_hidden,
            output_dim=1, n_layers=expert_layers, dropout=dropout,
        )

        # Time spent: input = trunk output
        self.time_expert = ExpertModule(
            input_dim=hidden_dim, hidden_dim=expert_hidden,
            output_dim=1, n_layers=expert_layers, dropout=dropout,
        )

        # ---- Move Head (deeper 3-layer, cross-feed input) ----
        # Cross-feed: trunk(hidden_dim) + wdl_h(expert_hidden)
        #           + mistake_h(expert_hidden) + time_h(expert_hidden)
        cross_feed_dim = hidden_dim + 3 * expert_hidden  # 640 with defaults
        move_head_in = cross_feed_dim + hidden_dim        # 896 with defaults
        self.move_head = nn.Sequential(
            nn.Linear(move_head_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
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
    # Core encoding helpers (identical to V1)
    # ------------------------------------------------------------------

    def _encode_possible_moves(
        self,
        possible_planes: torch.Tensor,
        possible_scalars: torch.Tensor,
        possible_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode each candidate move into an embedding vector.
        Only processes valid (unmasked) moves through the CNN to avoid
        wasting ~85% of GPU compute on zero-padded boards.

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
        valid = possible_mask.bool()                         # (B, M)

        # Pack only valid moves — skip ~85% of CNN work
        # Use tensor ops throughout so torch.compile can trace the graph
        flat_valid_planes = possible_planes[valid]           # (N_valid, C, H, W)
        flat_valid_scalars = possible_scalars[valid]         # (N_valid, scalar_dim)

        # CNN + scalar encoder on packed valid moves only
        flat_cnn = self.board_encoder(flat_valid_planes)     # (N_valid, cnn_channels)
        scalar_out = self.move_scalar_encoder(flat_valid_scalars)  # (N_valid, 32)
        combined = torch.cat([flat_cnn, scalar_out], dim=-1) # (N_valid, cnn_channels+32)
        proj = self.move_proj(combined)                      # (N_valid, hidden_dim)

        # Scatter back to (B, M, hidden_dim) with zeros for padding
        move_emb = torch.zeros(B, M, self.hidden_dim,
                               device=proj.device, dtype=proj.dtype)
        move_emb[valid] = proj
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
        Full forward pass with specialist expert heads and cross-feed fusion.

        Parameters
        ----------
        current_planes : (B, 23, 8, 8)
        possible_planes : (B, M, 23, 8, 8)
        possible_scalars : (B, M, 12)
        possible_mask : (B, M)   — 1.0 for valid moves, 0.0 for padding
        tabular : (B, 18)
        actual_idx : (B,) long or None
            Index of the move actually played within the possible-moves list.
            Required during training for:
              - win_prob_before masking
              - win_prob_after (selects actual move embedding)
            At inference time, pass None to get everything except win_prob_after.

        Returns
        -------
        dict with keys:
            move_logits     (B, M)
            mistake_prob    (B, 1)
            win_prob_before (B, 3)   — always present
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

        # --- Trunk fusion (single layer) ---
        fused = torch.cat([board_emb, tab_emb, attn_out], dim=-1)  # (B, 448)
        trunk_out = self.trunk(fused)                               # (B, hidden_dim)

        outputs: Dict[str, torch.Tensor] = {}

        # ============================================================
        # EXPERT HEADS (computed on trunk — mistake + time first)
        # ============================================================

        # Mistake expert
        mistake_hidden, mistake_logit = self.mistake_expert(trunk_out)
        outputs['mistake_prob'] = mistake_logit                          # (B, 1) — raw logits, sigmoid in loss

        # Time expert
        time_hidden, time_out = self.time_expert(trunk_out)
        outputs['time_spent'] = time_out                             # (B, 1)

        # ============================================================
        # WDL Before (MASKED trunk — same strategy as V1)
        # ============================================================
        # HEAD 3: Win probability before move
        # Uses the same trunk_out as other heads — no masking needed.
        wdl_before_hidden, wdl_before_logits = self.wdl_before_expert(trunk_out)
        outputs['win_prob_before'] = F.softmax(wdl_before_logits, dim=-1)  # (B, 3)

        # ============================================================
        # CROSS-FEED FUSION → Move Head (packed for efficiency)
        # ============================================================
        # Detach expert hiddens so move loss doesn't corrupt expert weights
        cross_feed = torch.cat([
            trunk_out,
            wdl_before_hidden.detach(),
            mistake_hidden.detach(),
            time_hidden.detach(),
        ], dim=-1)  # (B, hidden_dim + 3 * expert_hidden)

        # Pack valid moves only — skip ~86% of move_head compute
        valid = ~pad_mask  # (B, M)
        if valid.any():
            batch_idx, move_idx = torch.where(valid)  # (N_valid,)
            flat_move_emb = move_emb[batch_idx, move_idx]  # (N_valid, hidden_dim)
            flat_cross_feed = cross_feed[batch_idx]  # (N_valid, cross_feed_dim)
            
            flat_input = torch.cat([flat_cross_feed, flat_move_emb], dim=-1)  # (N_valid, ...)
            flat_scores = self.move_head(flat_input).squeeze(-1)  # (N_valid,)
            
            move_scores = torch.full((B, M), float('-inf'), device=move_emb.device, dtype=torch.float32)
            move_scores[batch_idx, move_idx] = flat_scores.to(move_scores.dtype)
        else:
            move_scores = torch.full((B, M), float('-inf'), device=move_emb.device, dtype=torch.float32)
        
        outputs['move_logits'] = move_scores

        # ============================================================
        # WDL After (sees actual move — NOT in cross-feed)
        # ============================================================
        if actual_idx is not None:
            safe_aidx = actual_idx.clamp(min=0)
            idx_exp = safe_aidx.unsqueeze(-1).expand(-1, self.hidden_dim)  # (B, hidden_dim)
            actual_move_emb = move_emb.gather(1, idx_exp.unsqueeze(1)).squeeze(1)  # (B, hidden_dim)
            wdl_after_input = torch.cat([trunk_out, actual_move_emb], dim=-1)  # (B, hidden_dim*2)
            _, wdl_after_logits = self.wdl_after_expert(wdl_after_input)
            outputs['win_prob_after'] = F.softmax(wdl_after_logits, dim=-1)  # (B, 3)

        return outputs


# ---------------------------------------------------------------------------
# Loss (unchanged from V1)
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
            if valid_mask.any():
                losses['mistake_prob'] = F.binary_cross_entropy_with_logits(
                    predictions['mistake_prob'].squeeze(-1)[valid_mask],
                    targets['is_mistake'][valid_mask],
                )
            else:
                losses['mistake_prob'] = torch.tensor(0.0, device=predictions['mistake_prob'].device)

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

    model = ChessMIMOModelV2(max_possible=M)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Dummy inputs
    current = torch.randn(B, 23, 8, 8)
    poss_planes = torch.randn(B, M, 23, 8, 8)
    poss_scalars = torch.randn(B, M, 12)
    poss_mask = torch.ones(B, M)
    poss_mask[:, 25:] = 0  # last 15 are padding
    tabular = torch.randn(B, 18)
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

    # Inference mode (no actual_idx)
    print("\n--- Inference mode (no actual_idx) ---")
    outputs_infer = model(current, poss_planes, poss_scalars, poss_mask, tabular)
    for k, v in outputs_infer.items():
        print(f"  {k:20s} {tuple(v.shape)}")
    assert 'win_prob_after' not in outputs_infer, "win_prob_after should not be present without actual_idx"
    print("  ✓ win_prob_after correctly absent")
