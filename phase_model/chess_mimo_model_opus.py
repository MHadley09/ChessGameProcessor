#!/usr/bin/env python3
"""
chess_mimo_model_opus.py — Multi-Input Multi-Output Chess Behavior Prediction Model

Architecture overview:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         INPUTS                                      │
    │  current_planes (B,47,8,8)  tabular (B,10)  possible_planes (B,M,47,8,8) │
    │                                             possible_scalars (B,M,6)│
    │                              game_phase (B,) int 0/1/2              │
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

Author: Sskeer (mimo_opus)
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
    6-residual-block CNN with SE attention for 47-plane chess positions.

    Accepts either (B, 47, 8, 8) or flattened batches from possible moves
    reshaped to (B*M, 47, 8, 8) externally.
    """

    def __init__(self, in_planes: int = 47, channels: int = 128, num_res_blocks: int = 6):
        super().__init__()
        # Stem: project 47 input planes to `channels`
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
        """x: (N, 47, 8, 8) → (N, channels)"""
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
        """x: (B, M, input_dim) → (B, M, output_dim)"""
        B, M, D = x.shape
        return self.net(x.reshape(B * M, D)).reshape(B, M, -1)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class ChessMIMOModelOpus(nn.Module):
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
        max_possible: int = 40,
        hidden_dim: int = 256,
        num_attn_heads: int = 4,
        dropout: float = 0.2,
        move_scalar_dim: int = 11,
    ):
        super().__init__()
        self.max_possible = max_possible
        self.hidden_dim = hidden_dim

        # ---- Encoders (shared CNN for current + possible boards) ----
        self.board_encoder = BoardEncoder(
            in_planes=47, channels=cnn_channels, num_res_blocks=num_res_blocks
        )
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_dim, output_dim=64, dropout=dropout * 0.5
        )
        self.move_scalar_encoder = PossibleMoveScalarEncoder(
            input_dim=move_scalar_dim, output_dim=32
        )

        # ---- Game-phase embedding (opening=0, middlegame=1, endgame=2) ----
        self.phase_embedding = nn.Embedding(3, 16)
        phase_emb_dim = 16

        # ---- Possible-move projection ----
        # CNN output (128) + scalar encoder output (32) → hidden_dim
        move_emb_dim = cnn_channels + 32
        self.move_proj = nn.Sequential(
            nn.Linear(move_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ---- Cross-attention: context queries, candidate-move keys/values ----
        # Query dimension = cnn_channels + tabular_out + phase_emb = 128 + 64 + 16 = 208
        self.query_proj = nn.Linear(cnn_channels + 64 + phase_emb_dim, hidden_dim)
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
        # Input: board_emb (128) + tabular (64) + phase (16) + attn_output (256) = 464
        fusion_in = cnn_channels + 64 + phase_emb_dim + hidden_dim
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

        # 2. Mistake probability (binary)
        self.mistake_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
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
    ) -> torch.Tensor:
        """
        Encode each candidate move into an embedding vector.

        Parameters
        ----------
        possible_planes : (B, M, 47, 8, 8)
        possible_scalars : (B, M, 11)

        Returns
        -------
        move_emb : (B, M, hidden_dim)
        """
        B, M, C, H, W = possible_planes.shape
        # Flatten batch for CNN
        flat_planes = possible_planes.reshape(B * M, C, H, W)
        flat_cnn = self.board_encoder(flat_planes)           # (B*M, cnn_channels)
        cnn_out = flat_cnn.reshape(B, M, -1)                 # (B, M, cnn_channels)
        # Encode scalars
        scalar_out = self.move_scalar_encoder(possible_scalars)  # (B, M, 32)
        # Combine and project
        combined = torch.cat([cnn_out, scalar_out], dim=-1)  # (B, M, cnn_channels+32)
        return self.move_proj(combined)                       # (B, M, hidden_dim)

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
        game_phase: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with built-in masking for win_prob_before.

        Parameters
        ----------
        current_planes : (B, 47, 8, 8)
        possible_planes : (B, M, 47, 8, 8)
        possible_scalars : (B, M, 11)
        possible_mask : (B, M)   — 1.0 for valid moves, 0.0 for padding
        tabular : (B, 18)
        actual_idx : (B,) long or None
            Index of the move actually played within the possible-moves list.
            Required during training for:
              - win_prob_before masking
              - win_prob_after (selects actual move embedding)
            At inference time, pass None to get everything except those two heads.
        game_phase : (B,) long or None
            Game phase index: 0=opening, 1=middlegame, 2=endgame.
            If None, defaults to middlegame (1) for all samples.

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

        # --- Encode current board ---
        board_emb = self.board_encoder(current_planes)       # (B, 128)

        # --- Encode tabular ---
        tab_emb = self.tabular_encoder(tabular)              # (B, 64)

        # --- Game phase embedding ---
        if game_phase is None:
            game_phase = torch.ones(B, dtype=torch.long, device=current_planes.device)
        phase_emb = self.phase_embedding(game_phase)         # (B, 16)

        # --- Encode all candidate moves ---
        move_emb = self._encode_possible_moves(possible_planes, possible_scalars)
        # move_emb: (B, M, hidden_dim)

        # --- Context vector (includes phase) ---
        context = torch.cat([board_emb, tab_emb, phase_emb], dim=-1)  # (B, 208)

        # Padding mask for attention (True = ignore)
        pad_mask = (possible_mask == 0)                      # (B, M)

        # --- Full cross-attention (all moves visible) ---
        attn_out = self._cross_attend(context, move_emb, pad_mask)  # (B, hidden_dim)
        attn_out = self.attn_gate(attn_out) * attn_out              # gated

        # --- Fusion ---
        fused = torch.cat([board_emb, tab_emb, phase_emb, attn_out], dim=-1)  # (B, 464)
        global_hidden = self.fusion(fused)                          # (B, 256)

        outputs: Dict[str, torch.Tensor] = {}

        # ============================================================
        # AUXILIARY HEADS (computed first for feedback into move head)
        # ============================================================

        # HEAD 2: Mistake probability
        outputs['mistake_prob'] = self.mistake_head(global_hidden)   # (B, 1)

        # HEAD 5: Time spent (log-scale)
        outputs['time_spent'] = self.time_head(global_hidden)        # (B, 1)

        # HEAD 3: Win probability before move (MASKED during training)
        if actual_idx is not None:
            # Training: mask the actual move from cross-attention
            mask_for_before = torch.ones(B, M, device=move_emb.device)
            mask_for_before.scatter_(1, actual_idx.unsqueeze(1), 0.0)
            pad_mask_before = pad_mask | (mask_for_before == 0)

            attn_out_masked = self._cross_attend(context, move_emb, pad_mask_before)
            attn_out_masked = self.attn_gate(attn_out_masked) * attn_out_masked  # gated
            fused_masked = torch.cat([board_emb, tab_emb, phase_emb, attn_out_masked], dim=-1)
            global_hidden_masked = self.fusion(fused_masked)
            wdl_before = F.softmax(
                self.wdl_before_head(global_hidden_masked), dim=-1
            )  # (B, 3)
        else:
            # Inference: no move to mask, use full context
            wdl_before = F.softmax(
                self.wdl_before_head(global_hidden), dim=-1
            )  # (B, 3)
        outputs['win_prob_before'] = wdl_before

        # ============================================================
        # HEAD 1: Move logits — with auxiliary feedback
        # ============================================================
        # Detach auxiliary predictions so move loss doesn't corrupt aux heads
        aux_feedback = torch.cat([
            outputs['mistake_prob'].detach(),   # (B, 1)
            outputs['time_spent'].detach(),     # (B, 1)
            wdl_before.detach(),                # (B, 3)
        ], dim=-1)  # (B, 5)

        global_exp = global_hidden.unsqueeze(1).expand(-1, M, -1)   # (B, M, 256)
        aux_exp = aux_feedback.unsqueeze(1).expand(-1, M, -1)       # (B, M, 5)
        move_input = torch.cat([global_exp, move_emb, aux_exp], dim=-1)  # (B, M, 517)
        move_scores = self.move_head(move_input).squeeze(-1)        # (B, M)
        move_scores = move_scores.masked_fill(pad_mask, float('-inf'))
        outputs['move_logits'] = move_scores

        # ============================================================
        # HEAD 4: Win probability after move (sees actual move)
        # ============================================================
        if actual_idx is not None:
            idx_exp = actual_idx.unsqueeze(-1).expand(-1, self.hidden_dim)  # (B, 256)
            actual_move_emb = move_emb.gather(1, idx_exp.unsqueeze(1)).squeeze(1)
            wdl_after_input = torch.cat([global_hidden, actual_move_emb], dim=-1)
            outputs['win_prob_after'] = F.softmax(
                self.wdl_after_head(wdl_after_input), dim=-1
            )  # (B, 3)

        return outputs


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class MIMOLossOpus(nn.Module):
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

        # 1. Move prediction — cross-entropy
        if 'move_logits' in predictions and 'move_idx' in targets:
            losses['move_logits'] = F.cross_entropy(
                predictions['move_logits'], targets['move_idx']
            )

        # 2. Mistake — binary cross-entropy
        if 'mistake_prob' in predictions and 'is_mistake' in targets:
            losses['mistake_prob'] = F.binary_cross_entropy(
                predictions['mistake_prob'].squeeze(-1),
                targets['is_mistake'],
            )

        # 3. WDL before — KL divergence (target is one-hot from game result)
        if 'win_prob_before' in predictions and 'win_prob_before' in targets:
            log_pred = torch.log(predictions['win_prob_before'].clamp(min=1e-8))
            losses['win_prob_before'] = F.kl_div(
                log_pred, targets['win_prob_before'], reduction='batchmean'
            )

        # 4. WDL after — KL divergence
        if 'win_prob_after' in predictions and 'win_prob_after' in targets:
            log_pred = torch.log(predictions['win_prob_after'].clamp(min=1e-8))
            losses['win_prob_after'] = F.kl_div(
                log_pred, targets['win_prob_after'], reduction='batchmean'
            )

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

    model = ChessMIMOModelOpus(max_possible=M)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Dummy inputs
    current = torch.randn(B, 47, 8, 8)
    poss_planes = torch.randn(B, M, 47, 8, 8)
    poss_scalars = torch.randn(B, M, 11)
    poss_mask = torch.ones(B, M)
    poss_mask[:, 25:] = 0  # last 15 are padding
    tabular = torch.randn(B, 18)
    actual_idx = torch.randint(0, 25, (B,))
    game_phase = torch.randint(0, 3, (B,))

    outputs = model(current, poss_planes, poss_scalars, poss_mask, tabular, actual_idx, game_phase)
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
    criterion = MIMOLossOpus()
    loss, ld = criterion(outputs, targets)
    print(f"\nTotal loss: {loss.item():.4f}")
    for k, v in ld.items():
        print(f"  {k:25s} {v:.4f}")
