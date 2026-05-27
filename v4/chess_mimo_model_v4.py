#!/usr/bin/env python3
"""
chess_mimo_model_v4.py — Single-CNN Specialist-Expert Chess Model

V4 = V2's expert-module architecture + V3's single-CNN optimization.

Same as V3: the CNN runs ONCE on the current position.  Per-move representations
come from indexing into the CNN feature map at from/to squares, combined with
learned square embeddings and per-move scalar features.

Same as V2: specialist ExpertModules for each auxiliary head, with cross-feed
fusion of their hidden states into a deeper 3-layer move head.

Speed impact: identical to V3 (~50× fewer CNN evaluations vs V2).

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
         │  ┌────▼──────────────────────────────┐
         │  │  MoveEncoder (lightweight)         │
         │  │  feat_map[from/to] + embeds + scalar│
         │  │  → (B, M, 256)                     │
         │  └────────────────────────┬───────────┘
         │                           │
         └──────────┬────────────────│
                    │                │
           ┌────────▼────────┐       │
           │  Context (192)  │       │
           │  board+tabular  │       │
           └────────┬────────┘       │
                    │    Cross-Attn   │
                    └────────┬───────┘
                    ┌────────▼────────┐
                    │  Trunk (1-layer)│
                    │  → 256 global   │
                    └────────┬────────┘
                             │
         ┌──────────────┬────┼────────────────┐
         │              │    │                │
    ┌────▼────┐   ┌─────▼──┐ │  ┌──────┐ ┌────▼────┐
    │WDL Before│   │Mistake│ │  │ Time │ │WDL After│
    │ Expert   │   │Expert │ │  │Expert│ │ Expert  │
    │→h:128    │   │→h:128 │ │  │→h:128│ │→h:128   │
    │→out:3    │   │→out:1 │ │  │→out:1│ │→out:3   │
    └────┬─────┘   └───┬───┘ │  └──┬───┘ └─────────┘
         │             │     │     │
         └──────┬──────┘     │     │
                │ (detach)   │     │ (detach)
                ▼            │     │
    ┌────────────────────────▼─────▼────────────┐
    │  Cross-Feed Fusion                         │
    │  [trunk(256) + wdl_h(128) + mis_h(128)     │
    │   + time_h(128)] = 640                     │
    │  + move_emb(256) per move = 896            │
    │  → 3-layer MLP → per-move score            │
    └────────────────────────────────────────────┘

Author: Sskeer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Building Blocks (shared with V3)
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid), nn.GELU(),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


class ResBlock(nn.Module):
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
    """6-ResBlock CNN returning both pooled embedding and spatial feature map."""

    def __init__(self, in_planes: int = 23, channels: int = 128, num_res_blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.GELU(),
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
        x = self.stem(x)
        x = self.tower(x)
        feature_map = x
        pooled = self.pool(x).flatten(1)
        return pooled, feature_map


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int = 18, hidden_dim: int = 64, output_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim), nn.GELU(),
        )
        self.out_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoveEncoder(nn.Module):
    """
    Lightweight per-move encoder.  Indexes the CNN feature map at from/to squares
    and combines with learned square embeddings, promotion type, and scalar features.
    """

    def __init__(self, cnn_channels: int = 128, scalar_dim: int = 12,
                 sq_embed_dim: int = 32, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.from_embed = nn.Embedding(64, sq_embed_dim)
        self.to_embed = nn.Embedding(64, sq_embed_dim)
        self.promo_embed = nn.Embedding(5, 8)
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, 48), nn.GELU(),
            nn.Linear(48, 32), nn.GELU(),
        )
        combine_dim = cnn_channels * 2 + sq_embed_dim * 2 + 8 + 32
        self.projection = nn.Sequential(
            nn.Linear(combine_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim

    def forward(self, from_sq, to_sq, promo, scalars, feature_map, mask):
        B, M = from_sq.shape
        C = self.cnn_channels

        from_idx = (7 - from_sq // 8) * 8 + from_sq % 8
        to_idx = (7 - to_sq // 8) * 8 + to_sq % 8
        flat_map = feature_map.reshape(B, C, 64)

        from_feat = flat_map.gather(2, from_idx.unsqueeze(1).expand(-1, C, -1)).permute(0, 2, 1)
        to_feat = flat_map.gather(2, to_idx.unsqueeze(1).expand(-1, C, -1)).permute(0, 2, 1)

        from_emb = self.from_embed(from_sq)
        to_emb = self.to_embed(to_sq)
        promo_emb = self.promo_embed(promo)
        scalar_emb = self.scalar_net(scalars.reshape(B * M, -1)).reshape(B, M, -1)

        combined = torch.cat([from_feat, to_feat, from_emb, to_emb, promo_emb, scalar_emb], dim=-1)
        proj = self.projection(combined.reshape(B * M, -1)).reshape(B, M, -1)
        return proj * mask.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Expert Module (from V2)
# ---------------------------------------------------------------------------

class ExpertModule(nn.Module):
    """Specialist encoder returning (hidden, output) for cross-feed fusion."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_d, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(dropout),
            ])
            in_d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        output = self.head(hidden)
        return hidden, output


# ---------------------------------------------------------------------------
# Main Model V4
# ---------------------------------------------------------------------------

class ChessMIMOModelV4(nn.Module):
    """
    V4 = V2 experts + V3 single-CNN.

    Expert modules for each auxiliary head with cross-feed hidden states
    into a deeper 3-layer move head, but the CNN runs only once per position.

    Parameters
    ----------
    cnn_channels : int
        Width of the residual CNN (default 128).
    num_res_blocks : int
        Number of residual blocks in the CNN (default 6).
    tabular_dim : int
        Number of scalar input features (default 18).
    max_possible : int
        Maximum candidate moves per position (default 220).
    hidden_dim : int
        Dimension of the global fused representation (default 256).
    num_attn_heads : int
        Heads in cross-attention (default 4).
    dropout : float
        Dropout rate (default 0.2).
    move_scalar_dim : int
        Number of scalar features per candidate move (default 12).
    expert_hidden : int
        Hidden dimension for expert modules (default 128).
    expert_layers : int
        Number of hidden layers per expert (default 2).
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

        # ---- Encoders ----
        self.board_encoder = BoardEncoder(
            in_planes=23, channels=cnn_channels, num_res_blocks=num_res_blocks
        )
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_dim, output_dim=64, dropout=dropout * 0.5
        )

        # ---- Move encoder (replaces CNN-per-move) ----
        self.move_encoder = MoveEncoder(
            cnn_channels=cnn_channels,
            scalar_dim=move_scalar_dim,
            sq_embed_dim=32,
            hidden_dim=hidden_dim,
            dropout=dropout * 0.5,
        )

        # ---- Cross-attention ----
        self.query_proj = nn.Linear(cnn_channels + 64, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_attn_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
        )

        # ---- Trunk (single-layer fusion — from V2) ----
        fusion_in = cnn_channels + 64 + hidden_dim  # 448
        self.trunk = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )

        # ---- Expert Modules (from V2) ----
        self.wdl_before_expert = ExpertModule(
            input_dim=hidden_dim, hidden_dim=expert_hidden,
            output_dim=3, n_layers=expert_layers, dropout=dropout,
        )
        self.wdl_after_expert = ExpertModule(
            input_dim=hidden_dim * 2, hidden_dim=expert_hidden,
            output_dim=3, n_layers=expert_layers, dropout=dropout,
        )
        self.mistake_expert = ExpertModule(
            input_dim=hidden_dim, hidden_dim=expert_hidden,
            output_dim=1, n_layers=expert_layers, dropout=dropout,
        )
        self.time_expert = ExpertModule(
            input_dim=hidden_dim, hidden_dim=expert_hidden,
            output_dim=1, n_layers=expert_layers, dropout=dropout,
        )

        # ---- Move Head (deeper 3-layer with cross-feed — from V2) ----
        cross_feed_dim = hidden_dim + 3 * expert_hidden  # 640
        move_head_in = cross_feed_dim + hidden_dim        # 896
        self.move_head = nn.Sequential(
            nn.Linear(move_head_in, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
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

    def _cross_attend(self, context, move_emb, key_padding_mask):
        query = self.query_proj(context).unsqueeze(1)
        attn_out, _ = self.cross_attn(query, move_emb, move_emb, key_padding_mask=key_padding_mask)
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
        Full forward pass — single CNN evaluation + expert cross-feed.

        Parameters
        ----------
        current_planes : (B, 23, 8, 8)
        possible_from_sq : (B, M) long
        possible_to_sq : (B, M) long
        possible_promo : (B, M) long
        possible_scalars : (B, M, 12)
        possible_mask : (B, M)
        tabular : (B, 18)
        actual_idx : (B,) long or None

        Returns
        -------
        dict with keys: move_logits, mistake_prob, win_prob_before,
                         win_prob_after (if actual_idx), time_spent
        """
        B = current_planes.shape[0]
        M = possible_from_sq.shape[1]

        # --- Clamp ---
        tabular = tabular.clamp(-1e4, 1e4)
        possible_scalars = possible_scalars.clamp(-1e4, 1e4)

        # --- Single CNN pass ---
        board_emb, feature_map = self.board_encoder(current_planes)

        # --- Tabular ---
        tab_emb = self.tabular_encoder(tabular)

        # --- Lightweight move encoding ---
        move_emb = self.move_encoder(
            possible_from_sq, possible_to_sq, possible_promo,
            possible_scalars, feature_map, possible_mask,
        )

        # --- Context + cross-attention ---
        context = torch.cat([board_emb, tab_emb], dim=-1)
        pad_mask = (possible_mask == 0)

        attn_out = self._cross_attend(context, move_emb, pad_mask)
        attn_out = self.attn_gate(attn_out) * attn_out

        # --- Trunk ---
        fused = torch.cat([board_emb, tab_emb, attn_out], dim=-1)
        trunk_out = self.trunk(fused)

        outputs: Dict[str, torch.Tensor] = {}

        # ============================================================
        # EXPERT HEADS
        # ============================================================
        mistake_hidden, mistake_logit = self.mistake_expert(trunk_out)
        outputs['mistake_prob'] = mistake_logit

        time_hidden, time_out = self.time_expert(trunk_out)
        outputs['time_spent'] = time_out

        wdl_before_hidden, wdl_before_logits = self.wdl_before_expert(trunk_out)
        outputs['win_prob_before'] = F.softmax(wdl_before_logits, dim=-1)

        # ============================================================
        # CROSS-FEED FUSION → Move Head
        # ============================================================
        cross_feed = torch.cat([
            trunk_out,
            wdl_before_hidden.detach(),
            mistake_hidden.detach(),
            time_hidden.detach(),
        ], dim=-1)  # (B, 640)

        cross_exp = cross_feed.unsqueeze(1).expand(-1, M, -1)
        full_input = torch.cat([cross_exp, move_emb], dim=-1)  # (B, M, 896)

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
            wdl_after_input = torch.cat([trunk_out, actual_move_emb], dim=-1)
            _, wdl_after_logits = self.wdl_after_expert(wdl_after_input)
            outputs['win_prob_after'] = F.softmax(wdl_after_logits, dim=-1)

        return outputs


# ---------------------------------------------------------------------------
# Loss (identical to V2/V3)
# ---------------------------------------------------------------------------

class MIMOLoss(nn.Module):
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

    model = ChessMIMOModelV4(max_possible=M)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"V4 Model parameters: {n_params:,}")

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

    # Inference mode
    print("\n--- Inference mode ---")
    outputs_infer = model(current, from_sq, to_sq, promo, poss_scalars, poss_mask, tabular)
    for k, v in outputs_infer.items():
        print(f"  {k:20s} {tuple(v.shape)}")
    assert 'win_prob_after' not in outputs_infer
    print("  ✓ win_prob_after correctly absent")
