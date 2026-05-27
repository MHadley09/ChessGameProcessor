#!/usr/bin/env python3
"""
chess_mimo_model_v3.py — Single-CNN Multi-Input Multi-Output Chess Model

V3 architecture eliminates the per-move CNN evaluation that dominated GPU time.
Instead of running the 6-ResBlock CNN on every possible resulting board (~54×
per batch position), V3 runs the CNN ONCE on the current position and indexes
into the spatial feature map at each move's from/to squares.

Speed impact: ~50× fewer CNN evaluations per batch.
    V1/V2:  B × M CNN forward passes  (e.g. 256 × 54 = 13,824)
    V3/V4:  B × 1 CNN forward passes  (e.g. 256)

Architecture overview:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         INPUTS                                      │
    │  current_planes (B,23,8,8)  tabular (B,18)                          │
    │  possible_from_sq (B,M)  possible_to_sq (B,M)  possible_promo (B,M) │
    │  possible_scalars (B,M,12)  possible_mask (B,M)                     │
    └────────┬──────────────────────┬─────────────────────────────────────┘
             │                      │
     ┌───────▼───────┐   ┌─────────▼──────┐
     │  CNN (ONCE)   │   │ Tabular MLP    │
     │  6 ResBlocks  │   │ 3-layer+LN     │
     │  + SE blocks  │   │ → 64-dim       │
     │  → pooled     │   └────────┬───────┘
     │    (B,128)    │            │
     │  → feat_map   │            │
     │    (B,128,8,8)│            │
     └───┬───────┬───┘            │
         │       │                │
         │  ┌────▼─────────────────────────────────────┐
         │  │  MoveEncoder (lightweight, per-move)      │
         │  │  ┌─────────────────────────────────────┐  │
         │  │  │ gather feat_map[from_sq] → (B,M,128)│  │
         │  │  │ gather feat_map[to_sq]   → (B,M,128)│  │
         │  │  │ learned from_embed       → (B,M,32) │  │
         │  │  │ learned to_embed         → (B,M,32) │  │
         │  │  │ promo_embed              → (B,M,8)  │  │
         │  │  │ scalar_net(scalars)      → (B,M,32) │  │
         │  │  │ concat → (B,M,360) → proj → (B,M,256)│ │
         │  │  └─────────────────────────────────────┘  │
         │  └────────────────────────────────┬──────────┘
         │                                   │
         └──────────┬────────────────────────│
                    │                        │
           ┌────────▼────────┐       ┌───────▼────────┐
           │  Context (192)  │       │  move_emb      │
           │  board+tabular  │       │  (B,M,256)     │
           └────────┬────────┘       └───────┬────────┘
                    │                        │
                    │    Cross-Attention      │
                    │    Q=context K,V=moves  │
                    └────────┬───────────────┘
                             │
                    ┌────────▼────────┐
                    │  Fusion (448→256)│
                    └────────┬────────┘
                             │
             ┌───────┬───────┼───────┬───────┐
             ▼       ▼       ▼       ▼       ▼
         ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
         │move  ││mistak││WDL   ││WDL   ││time  │
         │logits││e_prob││before││after ││spent │
         └──────┘└──────┘└──────┘└──────┘└──────┘

Masking strategy (same as V1):
    - move_logits: sees all possible moves, picks which one human played
    - mistake_prob: sees position + candidates, NOT which was played or result
    - win_prob_before: actual move NOT explicitly identified
    - win_prob_after: sees actual move embedding
    - time_spent: time_spent excluded from tabular features

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

    V3 change: returns BOTH the global-pooled embedding (B, channels) AND
    the spatial feature map (B, channels, 8, 8) so the MoveEncoder can index
    into specific squares.
    """

    def __init__(self, in_planes: int = 23, channels: int = 128, num_res_blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        layers = []
        for i in range(num_res_blocks):
            layers.append(ResBlock(channels))
            if (i + 1) % 2 == 0:
                layers.append(SqueezeExcitation(channels))
        self.tower = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, 23, 8, 8)

        Returns
        -------
        pooled : (B, channels)  — global average pooled embedding
        feature_map : (B, channels, 8, 8) — spatial feature map for move indexing
        """
        x = self.stem(x)
        x = self.tower(x)
        feature_map = x                         # (B, C, 8, 8)
        pooled = self.pool(x).flatten(1)         # (B, C)
        return pooled, feature_map


class TabularEncoder(nn.Module):
    """3-layer MLP with LayerNorm for scalar features (18 inputs)."""

    def __init__(self, input_dim: int = 18, hidden_dim: int = 64, output_dim: int = 64,
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


class MoveEncoder(nn.Module):
    """
    Lightweight per-move encoder using spatial CNN features + from/to squares + scalars.

    Replaces the expensive CNN-per-move approach from V1/V2.  For each candidate move:

    1. **Spatial CNN features** — index into the current position's CNN feature map
       at the from-square and to-square.  This tells the model *what piece* is moving
       (from-square features) and *what's at the destination* (to-square features)
       without re-running the CNN.

    2. **Learned square embeddings** — position-aware embeddings for from/to squares.

    3. **Promotion embedding** — small embedding for promotion piece type.

    4. **Scalar features** — per-move evaluation data (eval, WDL, nodes, depth,
       piece_val, is_capture, is_check, policy_prob, etc.)

    All concatenated and projected to hidden_dim for downstream use.

    Parameters
    ----------
    cnn_channels : int
        Channel width of the CNN feature map (default 128).
    scalar_dim : int
        Number of per-move scalar features (default 12).
    sq_embed_dim : int
        Dimension of learned square embeddings (default 32).
    hidden_dim : int
        Output dimension (default 256).
    dropout : float
        Dropout in projection (default 0.1).
    """

    def __init__(self, cnn_channels: int = 128, scalar_dim: int = 12,
                 sq_embed_dim: int = 32, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.cnn_channels = cnn_channels

        # Learned square embeddings
        self.from_embed = nn.Embedding(64, sq_embed_dim)
        self.to_embed = nn.Embedding(64, sq_embed_dim)

        # Promotion: 0=none, 1=knight, 2=bishop, 3=rook, 4=queen
        self.promo_embed = nn.Embedding(5, 8)

        # Scalar feature encoder
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, 48),
            nn.GELU(),
            nn.Linear(48, 32),
            nn.GELU(),
        )

        # Total input dim:
        #   from_cnn (cnn_channels) + to_cnn (cnn_channels)
        #   + from_embed (sq_embed_dim) + to_embed (sq_embed_dim)
        #   + promo (8) + scalar (32)
        combine_dim = cnn_channels * 2 + sq_embed_dim * 2 + 8 + 32
        self.projection = nn.Sequential(
            nn.Linear(combine_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim

    def forward(
        self,
        from_sq: torch.Tensor,
        to_sq: torch.Tensor,
        promo: torch.Tensor,
        scalars: torch.Tensor,
        feature_map: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        from_sq : (B, M) long — source squares (0-63, chess convention: 0=a1, 63=h8)
        to_sq : (B, M) long — destination squares (0-63)
        promo : (B, M) long — promotion piece (0=none, 1=n, 2=b, 3=r, 4=q)
        scalars : (B, M, scalar_dim) — per-move scalar features
        feature_map : (B, C, 8, 8) — CNN spatial feature map from current position
        mask : (B, M) float — 1.0 for valid moves, 0.0 for padding

        Returns
        -------
        move_emb : (B, M, hidden_dim)
        """
        B, M = from_sq.shape
        C = self.cnn_channels

        # --- 1. Index CNN feature map at from/to squares ---
        # Chess square convention: sq=0 → a1 (rank 0, file 0)
        # Feature map convention: [row=0] → rank 8 (top of board)
        # Conversion: feat_idx = (7 - sq // 8) * 8 + sq % 8
        from_idx = (7 - from_sq // 8) * 8 + from_sq % 8   # (B, M)
        to_idx = (7 - to_sq // 8) * 8 + to_sq % 8         # (B, M)

        flat_map = feature_map.reshape(B, C, 64)           # (B, C, 64)

        # gather: (B, C, 64) indexed by (B, 1, M).expand → (B, C, M) → permute → (B, M, C)
        from_feat = flat_map.gather(2, from_idx.unsqueeze(1).expand(-1, C, -1)).permute(0, 2, 1)
        to_feat = flat_map.gather(2, to_idx.unsqueeze(1).expand(-1, C, -1)).permute(0, 2, 1)

        # --- 2. Learned square embeddings ---
        from_emb = self.from_embed(from_sq)                # (B, M, sq_embed_dim)
        to_emb = self.to_embed(to_sq)                      # (B, M, sq_embed_dim)

        # --- 3. Promotion embedding ---
        promo_emb = self.promo_embed(promo)                 # (B, M, 8)

        # --- 4. Scalar features ---
        scalar_emb = self.scalar_net(
            scalars.reshape(B * M, -1)
        ).reshape(B, M, -1)                                # (B, M, 32)

        # --- 5. Combine and project ---
        combined = torch.cat([
            from_feat, to_feat,
            from_emb, to_emb,
            promo_emb,
            scalar_emb,
        ], dim=-1)                                          # (B, M, combine_dim)

        proj = self.projection(
            combined.reshape(B * M, -1)
        ).reshape(B, M, -1)                                # (B, M, hidden_dim)

        # --- 6. Mask padding moves to zero ---
        return proj * mask.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Main Model V3
# ---------------------------------------------------------------------------

class ChessMIMOModelV3(nn.Module):
    """
    Multi-Input Multi-Output model V3 — single CNN pass per position.

    Eliminates the dominant compute bottleneck of V1/V2 (CNN on every possible
    move's resulting board).  The CNN runs once; the MoveEncoder extracts
    per-move representations by indexing into the CNN's spatial feature map.

    Parameters
    ----------
    cnn_channels : int
        Width of the residual CNN (default 128).
    num_res_blocks : int
        Number of residual blocks in the CNN (default 6).
    tabular_dim : int
        Number of scalar input features (default 18).
    max_possible : int
        Maximum number of candidate moves per position (default 220).
    hidden_dim : int
        Dimension of the global fused representation (default 256).
    num_attn_heads : int
        Heads in the cross-attention over candidate moves (default 4).
    dropout : float
        Dropout rate (default 0.2).
    move_scalar_dim : int
        Number of scalar features per candidate move (default 12).
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

        # ---- Encoders ----
        self.board_encoder = BoardEncoder(
            in_planes=23, channels=cnn_channels, num_res_blocks=num_res_blocks
        )
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_dim, output_dim=64, dropout=dropout * 0.5
        )

        # ---- Move encoder (replaces CNN-per-move + PossibleMoveScalarEncoder + move_proj) ----
        self.move_encoder = MoveEncoder(
            cnn_channels=cnn_channels,
            scalar_dim=move_scalar_dim,
            sq_embed_dim=32,
            hidden_dim=hidden_dim,
            dropout=dropout * 0.5,
        )

        # ---- Cross-attention: context queries, candidate-move keys/values ----
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

        # ---- Fusion MLP ----
        fusion_in = cnn_channels + 64 + hidden_dim  # 448
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

        # 1. Move logits: per-move score
        #    Input: global_hidden (256) + move_emb (256) + aux_feedback (5) = 517
        self.move_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 2. Mistake probability (binary — outputs logits)
        self.mistake_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 3. Win prob before (WDL, 3-way)
        self.wdl_before_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

        # 4. Win prob after (WDL, 3-way)
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

    def _init_weights(self):
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
    def _cross_attend(
        self,
        context: torch.Tensor,
        move_emb: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_proj(context).unsqueeze(1)
        attn_out, _ = self.cross_attn(
            query, move_emb, move_emb, key_padding_mask=key_padding_mask
        )
        attn_out = self.attn_norm(attn_out.squeeze(1))
        return attn_out

    # ------------------------------------------------------------------
    def forward(
        self,
        current_planes: torch.Tensor,
        possible_from_sq: torch.Tensor,
        possible_to_sq: torch.Tensor,
        possible_promo: torch.Tensor,
        possible_scalars: torch.Tensor,
        possible_mask: torch.Tensor,
        tabular: torch.Tensor,
        actual_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass — single CNN evaluation.

        Parameters
        ----------
        current_planes : (B, 23, 8, 8)
        possible_from_sq : (B, M) long — source squares (0-63)
        possible_to_sq : (B, M) long — destination squares (0-63)
        possible_promo : (B, M) long — promotion piece (0=none, 1=n, 2=b, 3=r, 4=q)
        possible_scalars : (B, M, 12)
        possible_mask : (B, M) — 1.0 for valid, 0.0 for padding
        tabular : (B, 18)
        actual_idx : (B,) long or None

        Returns
        -------
        dict with keys:
            move_logits     (B, M)
            mistake_prob    (B, 1)
            win_prob_before (B, 3)
            win_prob_after  (B, 3)   — only if actual_idx provided
            time_spent      (B, 1)
        """
        B = current_planes.shape[0]
        M = possible_from_sq.shape[1]

        # --- Clamp scalar inputs ---
        tabular = tabular.clamp(-1e4, 1e4)
        possible_scalars = possible_scalars.clamp(-1e4, 1e4)

        # --- Encode current board (SINGLE CNN pass) ---
        board_emb, feature_map = self.board_encoder(current_planes)
        # board_emb: (B, 128),  feature_map: (B, 128, 8, 8)

        # --- Encode tabular ---
        tab_emb = self.tabular_encoder(tabular)              # (B, 64)

        # --- Encode all candidate moves (lightweight — no CNN per move) ---
        move_emb = self.move_encoder(
            possible_from_sq, possible_to_sq, possible_promo,
            possible_scalars, feature_map, possible_mask,
        )   # (B, M, hidden_dim)

        # --- Context vector ---
        context = torch.cat([board_emb, tab_emb], dim=-1)    # (B, 192)

        # Padding mask for attention (True = ignore)
        pad_mask = (possible_mask == 0)                      # (B, M)

        # --- Full cross-attention (all moves visible) ---
        attn_out = self._cross_attend(context, move_emb, pad_mask)
        attn_out = self.attn_gate(attn_out) * attn_out

        # --- Fusion ---
        fused = torch.cat([board_emb, tab_emb, attn_out], dim=-1)  # (B, 448)
        global_hidden = self.fusion(fused)                          # (B, 256)

        outputs: Dict[str, torch.Tensor] = {}

        # ============================================================
        # AUXILIARY HEADS
        # ============================================================
        outputs['mistake_prob'] = self.mistake_head(global_hidden)
        outputs['time_spent'] = self.time_head(global_hidden)

        wdl_before = F.softmax(
            self.wdl_before_head(global_hidden), dim=-1
        )
        outputs['win_prob_before'] = wdl_before

        # ============================================================
        # MOVE HEAD — with auxiliary feedback
        # ============================================================
        aux_feedback = torch.cat([
            outputs['mistake_prob'].detach().sigmoid(),
            outputs['time_spent'].detach(),
            wdl_before.detach(),
        ], dim=-1)  # (B, 5)

        global_exp = global_hidden.unsqueeze(1).expand(-1, M, -1)
        aux_exp = aux_feedback.unsqueeze(1).expand(-1, M, -1)
        full_input = torch.cat([global_exp, move_emb, aux_exp], dim=-1)  # (B, M, 517)

        flat_scores = self.move_head(full_input.reshape(B * M, -1)).squeeze(-1)
        move_scores = flat_scores.reshape(B, M).to(torch.float32)
        move_scores = move_scores.masked_fill(pad_mask, float('-inf'))
        outputs['move_logits'] = move_scores

        # ============================================================
        # WDL AFTER (sees actual move)
        # ============================================================
        if actual_idx is not None:
            safe_aidx = actual_idx.clamp(min=0)
            idx_exp = safe_aidx.unsqueeze(-1).expand(-1, self.hidden_dim)
            actual_move_emb = move_emb.gather(1, idx_exp.unsqueeze(1)).squeeze(1)
            wdl_after_input = torch.cat([global_hidden, actual_move_emb], dim=-1)
            outputs['win_prob_after'] = F.softmax(
                self.wdl_after_head(wdl_after_input), dim=-1
            )

        return outputs


# ---------------------------------------------------------------------------
# Loss (identical to V1)
# ---------------------------------------------------------------------------

class MIMOLoss(nn.Module):
    """Kendall multi-task loss with learnable log-variance weighting."""

    HEADS = ['move_logits', 'mistake_prob', 'win_prob_before', 'win_prob_after', 'time_spent']

    def __init__(self):
        super().__init__()
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in self.HEADS
        })

    def forward(self, predictions, targets):
        losses = {}

        if 'move_logits' in predictions and 'move_idx' in targets:
            losses['move_logits'] = F.cross_entropy(
                predictions['move_logits'], targets['move_idx'], ignore_index=-1,
            )

        if 'mistake_prob' in predictions and 'is_mistake' in targets:
            valid_mask = targets['move_idx'] >= 0
            raw_bce = F.binary_cross_entropy_with_logits(
                predictions['mistake_prob'].squeeze(-1),
                targets['is_mistake'], reduction='none',
            )
            masked_bce = raw_bce * valid_mask.float()
            losses['mistake_prob'] = masked_bce.sum() / valid_mask.float().sum().clamp(min=1.0)

        if 'win_prob_before' in predictions and 'win_prob_before' in targets:
            log_pred = torch.log(predictions['win_prob_before'].clamp(min=1e-8))
            losses['win_prob_before'] = -(targets['win_prob_before'] * log_pred).sum(-1).mean()

        if 'win_prob_after' in predictions and 'win_prob_after' in targets:
            log_pred = torch.log(predictions['win_prob_after'].clamp(min=1e-8))
            losses['win_prob_after'] = -(targets['win_prob_after'] * log_pred).sum(-1).mean()

        if 'time_spent' in predictions and 'time_spent_log' in targets:
            losses['time_spent'] = F.huber_loss(
                predictions['time_spent'].squeeze(-1),
                targets['time_spent_log'], delta=2.0,
            )

        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for name, loss_val in losses.items():
            log_var = self.log_vars[name]
            precision = torch.exp(-log_var)
            total = total + precision * loss_val + log_var

        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict['total'] = total.item()
        for name in losses:
            loss_dict[f'w_{name}'] = torch.exp(-self.log_vars[name]).item()
        return total, loss_dict


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(42)
    B, M = 4, 40

    model = ChessMIMOModelV3(max_possible=M)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"V3 Model parameters: {n_params:,}")

    # Dummy inputs (no possible_planes!)
    current = torch.randn(B, 23, 8, 8)
    from_sq = torch.randint(0, 64, (B, M))
    to_sq = torch.randint(0, 64, (B, M))
    promo = torch.zeros(B, M, dtype=torch.long)
    poss_scalars = torch.randn(B, M, 12)
    poss_mask = torch.ones(B, M)
    poss_mask[:, 25:] = 0
    tabular = torch.randn(B, 18)
    actual_idx = torch.randint(0, 25, (B,))

    outputs = model(current, from_sq, to_sq, promo, poss_scalars, poss_mask, tabular, actual_idx)
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
