#!/usr/bin/env python3
"""
chess_mimo_model_v5.py — Single-CNN Specialist-Expert Chess Model
                         + Contrastive Move Expert
                         + Phase-Gated Experts
                         + Attention Move Head

V5 = V4 + four new capabilities (each independently toggleable):

  1. ContrastiveEncoder: 2-layer MLP producing 64-dim per-move "preference
     embeddings" trained via self-referential triplet margin loss.  These
     embeddings are concatenated into the move head input (+64 dims).

  2. PhaseGatedExperts: A shared PhaseEncoder detects game phase (opening /
     middlegame / endgame) from trunk_out, producing soft 3-way weights.
     Each auxiliary head (mistake, time, wdl) is replaced by 3 phase-specific
     sub-experts whose outputs are blended by those weights.  Controlled by
     `use_phase_experts` flag for clean ablation.

  3. AttentionMoveHead: A registered move-head option ('attention') that runs
     a small transformer over legal moves before scoring — captures inter-move
     relationships.  Selected via --move-head-ver attention or swap_move_head().

  4. FiLM Conditioning: Feature-wise Linear Modulation on the CNN trunk.
     A small MLP converts raw tabular features (Elo, time control, etc.)
     into per-ResBlock (gamma, beta) pairs that scale and shift the residual
     branch output.  This lets the same CNN extract skill-level-appropriate
     board features — a 1200 and a 2500 seeing the same position get
     different trunk representations.  Controlled by `use_film` flag.
     ~105K additional params (~3.7% of base model).

  5. TacticalEnrichment: Frozen pre-trained TacticalPreprocessor injects
     384-dim gated theme+opening embeddings into the trunk via a learned
     residual projection.  Off by default (`use_tactical_enrichment=False`).
     Requires a trained `tactical_models.TacticalPreprocessor` checkpoint.

     a small transformer over legal moves before scoring — captures inter-move
     relationships.  Selected via --move-head-ver attention or swap_move_head().

Architecture overview (all features enabled):
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         INPUTS                                      │
    │  current_planes (B,23,8,8)  tabular (B,20)                          │
    │  possible_from_sq (B,M)  possible_to_sq (B,M)  possible_promo (B,M) │
    │  possible_scalars (B,M,13)  possible_mask (B,M)                     │
    └────────┬──────────────────────┬─────────────────────────────────────┘
             │                      │
             │               ┌──────▼───────┐
             │               │ FiLM Cond.   │ (use_film=True)
             │               │ tabular→MLP  │
             │               │ → per-block  │
             │               │   (γ,β)×6    │
             │               └──────┬───────┘
             │                      │
     ┌───────▼──────────────────────▼─┐   ┌─────────────────┐
     │  CNN (ONCE)                     │   │ Tabular MLP     │
     │  6 ResBlocks + FiLM modulation  │   │ 3-layer+LN      │
     │  + SE blocks                    │   │ → 64-dim        │
     │  → pooled (B,128)              │   └────────┬────────┘
     │  → feat_map (B,128,8,8)        │            │
     └───┬───────┬─────────────────────┘            │
         │       │                │
         │  ┌────▼──────────────────────────────┐
         │  │  MoveEncoder (lightweight)         │
         │  │  feat_map[from/to] + embeds + scalar│
         │  │  → (B, M, 256) = move_emb          │
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
                    ┌────────▼────────┐
                    │  PhaseEncoder   │  (optional, use_phase_experts=True)
                    │  → (B, 3) soft  │
                    └────────┬────────┘
                             │
         ┌──────────────┬────┼────────────────┐
         │              │    │                │
    ┌────▼────┐   ┌─────▼──┐ │  ┌──────┐ ┌────▼──────┐
    │WDL      │   │Mistake│ │  │ Time │ │Contrastive│
    │PhaseGtd │   │PhsGtd│ │  │PhsGtd│ │Encoder    │
    │3×Expert │   │3×Exp │ │  │3×Exp │ │→64-dim    │
    │→h:128   │   │→h:128│ │  │→h:128│ │per move   │
    └────┬────┘   └───┬──┘ │  └──┬───┘ └────┬──────┘
         │            │    │     │           │
         └─────┬──────┘    │     │           │
               │ (detach)  │     │           │
               ▼           │     │           │
    ┌──────────────────────▼─────▼───────────▼────┐
    │  Cross-Feed Fusion                            │
    │  [trunk(256) + wdl_h(128) + mis_h(128)        │
    │   + time_h(128)] = 640                        │
    │  + move_emb(256) + contrastive_embed(64) = 960│
    │  → MLP or Attention Transformer → move scores │
    └─────────────────────────────────────────────────┘

Author: Sskeer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Input sanitization (fix #5 + #7)
# ---------------------------------------------------------------------------
# Stored shards from an older writer can contain sentinel spikes (e.g. -42069
# for missing/mate evals) and occasional non-finite values. Every feature slot
# that can carry such a spike is eval-derived, where 0.0 (even position) is the
# correct neutral. All other slots (WDL on 0-100, 0/1 flags, elo/3000, etc.) are
# bounded well under SENTINEL_THRESHOLD by construction, so this is a no-op for
# clean data and only repairs poisoned samples.
SENTINEL_THRESHOLD = 1000.0
SANITIZE_BOUND = 50.0


def sanitize_features(x: torch.Tensor,
                      threshold: float = SENTINEL_THRESHOLD,
                      bound: float = SANITIZE_BOUND) -> torch.Tensor:
    """Map non-finite + sentinel-magnitude entries to neutral 0.0, then bound.

    Behavior-neutral for in-distribution features (|value| << threshold).
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.where(x.abs() > threshold, torch.zeros_like(x), x)
    return x.clamp(-bound, bound)


# ---------------------------------------------------------------------------
# Building Blocks (shared with V3/V4)
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

    def forward(self, x: torch.Tensor,
                film: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                ) -> torch.Tensor:
        residual = x
        out = F.gelu(self.bn1(x))
        out = self.conv1(out)
        out = F.gelu(self.bn2(out))
        out = self.conv2(out)
        if film is not None:
            gamma, beta = film  # each (B, C)
            out = gamma[:, :, None, None] * out + beta[:, :, None, None]
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

    def forward(self, x: torch.Tensor,
                film_params: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        if film_params is None:
            x = self.tower(x)  # fast path — no FiLM
        else:
            res_idx = 0
            for module in self.tower:
                if isinstance(module, ResBlock):
                    x = module(x, film=film_params[res_idx])
                    res_idx += 1
                else:
                    x = module(x)  # SE block
        feature_map = x
        pooled = self.pool(x).flatten(1)
        return pooled, feature_map


class FiLMConditioner(nn.Module):
    """Generate per-ResBlock (gamma, beta) modulation from player context.

    FiLM (Feature-wise Linear Modulation) conditions the CNN trunk on
    player features (Elo, time control, etc.) so the same architecture
    extracts skill-level-appropriate board features.  A 1200 and a 2500
    see the same position but the trunk learns different feature emphasis
    for each.

    Initialized near identity (gamma=1, beta=0) so an untrained FiLM
    conditioner has zero effect on the trunk output.

    Parameter overhead: ~105K params for 6 blocks / 128 channels / hidden 64
    (about 3.7% of a 2.8M-param model).
    """

    def __init__(self, input_dim: int = 20, num_blocks: int = 6,
                 channels: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Per-block (gamma, beta) projections
        self.film_projections = nn.ModuleList([
            nn.Linear(hidden_dim, 2 * channels)
            for _ in range(num_blocks)
        ])

        # Initialize projections to zero so gamma starts at 1.0, beta at 0.0
        for proj in self.film_projections:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, player_features: torch.Tensor
                ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            player_features: (B, input_dim) — raw tabular features
        Returns:
            list of (gamma, beta) tuples, one per ResBlock.
            gamma: (B, channels), centered around 1.0
            beta:  (B, channels), centered around 0.0
        """
        h = self.encoder(player_features)           # (B, hidden_dim)
        films: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for proj in self.film_projections:
            gb = proj(h)                             # (B, 2*channels)
            gamma, beta = gb.chunk(2, dim=-1)        # each (B, channels)
            gamma = gamma + 1.0                      # identity init
            films.append((gamma, beta))
        return films


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, output_dim: int = 64,
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

    def __init__(self, cnn_channels: int = 128, scalar_dim: int = 13,
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

    def forward(self, x: torch.Tensor, phase_weights: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """phase_weights is accepted but ignored for API compatibility."""
        hidden = self.backbone(x)
        output = self.head(hidden)
        return hidden, output


# ---------------------------------------------------------------------------
# Phase-Gated Expert (NEW in V5)
# ---------------------------------------------------------------------------

class PhaseEncoder(nn.Module):
    """Detects game phase (opening/middlegame/endgame) and produces soft blend weights.

    The phase signal is already encoded in trunk_out because the CNN sees
    halfmove clock (plane 22), piece planes (0-11), and tabular has
    material/clock info.  The PhaseEncoder learns to detect phase from
    these features.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 64,
                 num_phases: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_phases),  # raw logits
        )
        self.num_phases = num_phases

    def forward(self, trunk_out: torch.Tensor) -> torch.Tensor:
        """Returns (B, num_phases) softmax weights."""
        return F.softmax(self.net(trunk_out), dim=-1)


class PhaseGatedExpert(nn.Module):
    """Three phase-specific sub-experts blended by phase weights.

    Each sub-expert is a standard ExpertModule.  The phase weights from the
    shared PhaseEncoder determine how to blend their hidden states and outputs.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_phases: int = 3, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.phase_experts = nn.ModuleList([
            ExpertModule(input_dim, hidden_dim, output_dim,
                         n_layers=n_layers, dropout=dropout)
            for _ in range(num_phases)
        ])
        self.num_phases = num_phases

    def forward(self, x: torch.Tensor, phase_weights: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) — trunk representation
            phase_weights: (B, num_phases) — soft blend from PhaseEncoder.
                           If None, uses uniform weights (fallback).

        Returns:
            (hidden, output) — blended across phases.
        """
        hiddens = []
        outputs = []
        for expert in self.phase_experts:
            h, o = expert(x)
            hiddens.append(h)
            outputs.append(o)

        # Stack: (B, num_phases, hidden_dim) and (B, num_phases, output_dim)
        hiddens = torch.stack(hiddens, dim=1)
        outputs = torch.stack(outputs, dim=1)

        if phase_weights is None:
            phase_weights = torch.ones(
                x.shape[0], self.num_phases,
                device=x.device, dtype=x.dtype,
            ) / self.num_phases

        # Blend: (B, num_phases, 1) * (B, num_phases, dim) → sum over phases
        w = phase_weights.unsqueeze(-1)  # (B, num_phases, 1)
        blended_hidden = (w * hiddens).sum(dim=1)  # (B, hidden_dim)
        blended_output = (w * outputs).sum(dim=1)  # (B, output_dim)

        return blended_hidden, blended_output


# ---------------------------------------------------------------------------
# Contrastive Move Expert (NEW in V5)
# ---------------------------------------------------------------------------

class ContrastiveEncoder(nn.Module):
    """
    Per-move contrastive encoder producing preference embeddings.

    Takes move_emb (B, M, input_dim) and outputs contrastive embeddings
    (B, M, output_dim) for use in triplet margin loss and as additional
    input features to the move head.

    Architecture: 2-layer MLP with LayerNorm and GELU.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 output_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self.output_dim = output_dim

    def forward(self, move_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            move_emb: (B, M, input_dim) — per-move representations

        Returns:
            (B, M, output_dim) — contrastive preference embeddings
        """
        B, M, D = move_emb.shape
        flat = self.net(move_emb.reshape(B * M, D))
        return flat.reshape(B, M, -1)


# ---------------------------------------------------------------------------
# Attention Move Head (NEW in V5)
# ---------------------------------------------------------------------------

class AttentionMoveHead(nn.Module):
    """
    Small transformer over legal moves.  Each move attends to other legal moves
    before being scored — captures inter-move relationships like
    "I play Nf3 because I rejected e4."

    Unlike the default MLP move head (which processes each move independently),
    this head takes (B, M, input_dim) and returns (B, M) scores.  The model's
    forward() detects this via isinstance and passes the full (B, M, dim) tensor
    plus the padding mask.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, M, input_dim) — per-move features (cross_feed + move_emb + contrastive)
            padding_mask: (B, M) bool — True where padded (same convention as pad_mask)

        Returns:
            (B, M) — per-move scores (before masking by caller)
        """
        h = self.input_proj(x)           # (B, M, hidden_dim)
        h = h + self.pos_encoding        # broadcast learnable position encoding

        # Self-attention with padding mask
        h = self.transformer(h, src_key_padding_mask=padding_mask)  # (B, M, hidden_dim)

        # Per-move scores
        scores = self.score_proj(h).squeeze(-1)  # (B, M)
        return scores


class PhaseGatedMoveHead(nn.Module):
    """Three phase-specific move heads blended by phase weights.

    Wraps any move head type (MLP or AttentionMoveHead).  The PhaseEncoder's
    soft weights (B, num_phases) are broadcast across the M move dimension so
    each position gets phase-specialised move scoring.
    """

    def __init__(self, head_factory, num_phases: int = 3):
        """
        Args:
            head_factory: callable() -> nn.Module producing one move head instance.
            num_phases:   number of phase-specific sub-heads (default 3).
        """
        super().__init__()
        self.phase_heads = nn.ModuleList([head_factory() for _ in range(num_phases)])
        self.num_phases = num_phases

    def forward(self, x: torch.Tensor,
                pad_mask: torch.Tensor,
                phase_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:             (B, M, input_dim)
            pad_mask:      (B, M) bool — True where padded
            phase_weights: (B, num_phases) from PhaseEncoder.  If None, uniform.

        Returns:
            (B, M) — blended per-move scores (before final -inf masking by caller)
        """
        B, M, D = x.shape

        scores_list = []
        for head in self.phase_heads:
            if isinstance(head, AttentionMoveHead):
                s = head(x, pad_mask)                       # (B, M)
            else:
                flat = head(x.reshape(B * M, D)).squeeze(-1)
                s = flat.reshape(B, M)
            scores_list.append(s)

        # (B, num_phases, M)
        all_scores = torch.stack(scores_list, dim=1)

        if phase_weights is None:
            phase_weights = torch.ones(
                B, self.num_phases, device=x.device, dtype=x.dtype,
            ) / self.num_phases

        # (B, num_phases, 1) * (B, num_phases, M) → sum → (B, M)
        w = phase_weights.unsqueeze(-1)                     # (B, num_phases, 1)
        blended = (w * all_scores).sum(dim=1)               # (B, M)
        return blended


# ---------------------------------------------------------------------------
# Expert & Move Head Registries
# ---------------------------------------------------------------------------

_EXPERT_REGISTRY: Dict[str, dict] = {}
_MOVE_HEAD_REGISTRY: Dict[str, dict] = {}


def register_expert(name: str, factory, hidden_dim: int):
    """Register an expert architecture.

    factory(input_dim, output_dim, dropout) -> nn.Module
    forward(x, phase_weights=None) must return (hidden, output)
    where hidden.shape[-1] == hidden_dim.
    """
    _EXPERT_REGISTRY[name] = {'factory': factory, 'hidden_dim': hidden_dim}


def register_move_head(name: str, factory):
    """Register a move-head architecture.

    factory(input_dim, hidden_dim, dropout) -> nn.Module
    Standard heads: forward(B*M, input_dim) -> (B*M, 1)
    Attention heads: forward(B, M, input_dim, padding_mask) -> (B, M)
    """
    _MOVE_HEAD_REGISTRY[name] = {'factory': factory}


def list_experts() -> Dict[str, int]:
    """Return {name: hidden_dim} for all registered experts."""
    return {k: v['hidden_dim'] for k, v in _EXPERT_REGISTRY.items()}


def list_move_heads() -> list:
    """Return names of all registered move heads."""
    return list(_MOVE_HEAD_REGISTRY.keys())


# ---- Built-in experts (V4 compatible) ----

register_expert('default',
    lambda in_d, out_d, dropout: ExpertModule(in_d, 128, out_d, n_layers=2, dropout=dropout),
    hidden_dim=128)

register_expert('deep_4L',
    lambda in_d, out_d, dropout: ExpertModule(in_d, 128, out_d, n_layers=4, dropout=dropout),
    hidden_dim=128)

register_expert('wide_256',
    lambda in_d, out_d, dropout: ExpertModule(in_d, 256, out_d, n_layers=2, dropout=dropout),
    hidden_dim=256)

register_expert('deep_wide',
    lambda in_d, out_d, dropout: ExpertModule(in_d, 256, out_d, n_layers=4, dropout=dropout),
    hidden_dim=256)

# Wider single expert (V5 widehead) — ~3.4x a former phase sub-expert.
# Used by the 'v5-widehead' preset in place of phase-gated experts.
register_expert('wide_384',
    lambda in_d, out_d, dropout: ExpertModule(in_d, 384, out_d, n_layers=2, dropout=dropout),
    hidden_dim=384)

# ---- Contrastive expert (V5) ----

register_expert('contrastive_default',
    lambda in_d, out_d, dropout: ContrastiveEncoder(in_d, 128, out_d, dropout=dropout),
    hidden_dim=64)

# ---- Phase-gated experts (V5) ----

register_expert('phase_gated_default',
    lambda in_d, out_d, dropout: PhaseGatedExpert(in_d, 128, out_d, num_phases=3, n_layers=2, dropout=dropout),
    hidden_dim=128)

register_expert('phase_gated_deep',
    lambda in_d, out_d, dropout: PhaseGatedExpert(in_d, 128, out_d, num_phases=3, n_layers=4, dropout=dropout),
    hidden_dim=128)

register_expert('phase_gated_wide',
    lambda in_d, out_d, dropout: PhaseGatedExpert(in_d, 256, out_d, num_phases=3, n_layers=2, dropout=dropout),
    hidden_dim=256)


# ---- Built-in MLP move heads (V4 compatible) ----

register_move_head('default',
    lambda in_d, hid_d, dropout: nn.Sequential(
        nn.Linear(in_d, hid_d), nn.LayerNorm(hid_d), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hid_d, hid_d // 2), nn.GELU(),
        nn.Linear(hid_d // 2, 1),
    ))

register_move_head('deep_4L',
    lambda in_d, hid_d, dropout: nn.Sequential(
        nn.Linear(in_d, hid_d), nn.LayerNorm(hid_d), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hid_d, hid_d), nn.LayerNorm(hid_d), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hid_d, hid_d // 2), nn.GELU(),
        nn.Linear(hid_d // 2, 1),
    ))

register_move_head('wide_512',
    lambda in_d, hid_d, dropout: nn.Sequential(
        nn.Linear(in_d, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(512, hid_d), nn.GELU(),
        nn.Linear(hid_d, 1),
    ))

# Single wide MLP move head (V5 widehead) — ~2.5x a former phase sub-head.
# Used by the 'v5-widehead' preset (use_phase_experts=False).
register_move_head('mlp_640',
    lambda in_d, hid_d, dropout: nn.Sequential(
        nn.Linear(in_d, 640), nn.LayerNorm(640), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(640, 320), nn.GELU(),
        nn.Linear(320, 1),
    ))

# Larger single MLP move head (~1.85M @ in_d=2016) — current 'v5-widehead'
# default. ~3x a former phase sub-head.
register_move_head('mlp_768',
    lambda in_d, hid_d, dropout: nn.Sequential(
        nn.Linear(in_d, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(768, 384), nn.GELU(),
        nn.Linear(384, 1),
    ))

# ---- Attention move heads (V5) ----

register_move_head('attention',
    lambda in_d, hid_d, dropout: AttentionMoveHead(
        input_dim=in_d, hidden_dim=hid_d, num_heads=4, num_layers=2, dropout=dropout,
    ))

register_move_head('attention_deep',
    lambda in_d, hid_d, dropout: AttentionMoveHead(
        input_dim=in_d, hidden_dim=hid_d, num_heads=4, num_layers=3, dropout=dropout,
    ))


# ---------------------------------------------------------------------------
# Main Model V5
# ---------------------------------------------------------------------------

class ChessMIMOModelV5(nn.Module):
    """
    V5 = V4 + ContrastiveEncoder + PhaseGatedExperts + AttentionMoveHead + FiLM.

    All four features are independently toggleable for clean ablation:
      - contrastive_embed_dim=0 disables ContrastiveEncoder
      - use_phase_experts=False falls back to V4-style single experts
      - move_head_ver='default' uses the standard MLP move head
      - use_film=False disables FiLM Elo conditioning on the CNN trunk

    Preset Configurations
    ---------------------
    Use ``ChessMIMOModelV5.from_preset(name)`` to instantiate a standard
    configuration.  Constructor defaults match the **V5** (default) preset.

    +--------------+--------+---------+---------------------------------------------+
    | Preset       |  ~MLP  | ~w/Attn | Notes                                       |
    +--------------+--------+---------+---------------------------------------------+
    | v5-minimal   |  3.9M  |    —    | Ablation / rapid iteration baseline         |
    | v5 (default) |  9.7M  | 14.2M   | Default training config                     |
    | v5-large     | 19.5M  | 29.1M   | Full capacity; target for Maia comparison   |
    +--------------+--------+---------+---------------------------------------------+

    Parameters
    ----------
    cnn_channels : int
        Width of the residual CNN (default 192).
    num_res_blocks : int
        Number of residual blocks in the CNN (default 8).
    tabular_dim : int
        Number of scalar input features (default 20).
    max_possible : int
        Maximum candidate moves per position (default 220).
    hidden_dim : int
        Dimension of the global fused representation (default 384).
    num_attn_heads : int
        Heads in cross-attention (default 4).
    dropout : float
        Dropout rate (default 0.2).
    move_scalar_dim : int
        Number of scalar features per candidate move (default 13).
    expert_hidden : int
        Hidden dimension for expert modules (default 160).
    expert_layers : int
        Number of hidden layers per expert (default 2).
    contrastive_embed_dim : int
        Dimension of contrastive embeddings (default 96, 0 to disable).
    contrastive_hidden_dim : int
        Hidden layer size in contrastive encoder (default 192).
    contrastive_margin : float
        Triplet margin for contrastive loss (default 1.0).
    use_phase_experts : bool
        Enable phase-gated experts for auxiliary heads (default True).
    phase_hidden_dim : int
        Hidden size in PhaseEncoder (default 96).
    num_phases : int
        Number of game phases (default 3 = opening/middlegame/endgame).
    use_film : bool
        Enable FiLM (Feature-wise Linear Modulation) conditioning on the
        CNN trunk.  Raw tabular features (including Elo) modulate each
        ResBlock's output via learned (gamma, beta) per block (default True).
    film_hidden_dim : int
        Hidden layer size in the FiLM conditioner MLP (default 96).
    use_tactical_enrichment : bool
        Enable frozen tactical/opening preprocessor trunk enrichment
        (default False).
    tactical_preprocessor_path : str or None
        Path to a trained TacticalPreprocessor checkpoint (.pt).  Used
        only on first build; weights are saved in the V5 checkpoint.
    tactical_preprocessor_config : dict or None
        TacticalPreprocessor constructor kwargs (for checkpoint reload).
    """

    # ================================================================
    # PRESET CONFIGURATIONS
    # ================================================================
    PRESETS = {
        'v5-minimal': {
            'cnn_channels': 128,
            'num_res_blocks': 6,
            'hidden_dim': 256,
            'expert_hidden': 128,
            'contrastive_embed_dim': 64,
            'contrastive_hidden_dim': 128,
            'film_hidden_dim': 64,
            'phase_hidden_dim': 64,
            'move_head_ver': 'attention_deep',
            'sq_embed_dim': 32,
        },
        'v5': {
            # Default — constructor defaults match this preset.
            'cnn_channels': 192,
            'num_res_blocks': 8,
            'hidden_dim': 384,
            'expert_hidden': 160,
            'contrastive_embed_dim': 96,
            'contrastive_hidden_dim': 192,
            'film_hidden_dim': 96,
            'phase_hidden_dim': 96,
            'move_head_ver': 'attention_deep',
            'sq_embed_dim': 48,
        },
        'v5-large': {
            'cnn_channels': 256,
            'num_res_blocks': 10,
            'hidden_dim': 512,
            'expert_hidden': 192,
            'contrastive_embed_dim': 128,
            'contrastive_hidden_dim': 256,
            'film_hidden_dim': 128,
            'phase_hidden_dim': 128,
            'move_head_ver': 'attention_deep',
            'sq_embed_dim': 48,
        },
        'v5-widehead': {
            # ~10.2M. Phase experts replaced by wider single heads after the
            # PhaseEncoder collapsed (opening~0.88) — the gate averaged 3
            # redundant sub-heads into ~1, so capacity is reallocated to fewer,
            # larger heads + one extra res block. No phase gating.
            'cnn_channels': 192,
            'num_res_blocks': 9,
            'hidden_dim': 384,
            'expert_hidden': 384,          # 3x former phase sub-expert width
            'mistake_expert_ver': 'wide_384',
            'time_expert_ver': 'wide_384',
            'wdl_expert_ver': 'wide_384',
            'move_head_ver': 'mlp_768',     # single wide MLP move head (~1.85M)
            'use_phase_experts': False,     # disable phase gating entirely
            'contrastive_embed_dim': 96,
            'contrastive_hidden_dim': 192,
            'film_hidden_dim': 96,
            'sq_embed_dim': 48,
        },
    }

    @classmethod
    def from_preset(cls, name: str = 'v5', **overrides):
        """Create a model from a named preset, with optional overrides.

        Example::

            model = ChessMIMOModelV5.from_preset('v5-large', move_head_ver='attention_deep')
        """
        if name not in cls.PRESETS:
            raise ValueError(
                f"Unknown preset '{name}'. Choose from: {list(cls.PRESETS.keys())}"
            )
        config = dict(cls.PRESETS[name])
        config.update(overrides)
        return cls(**config)

    def __init__(
        self,
        cnn_channels: int = 192,
        num_res_blocks: int = 8,
        tabular_dim: int = 20,
        max_possible: int = 220,
        hidden_dim: int = 384,
        num_attn_heads: int = 4,
        dropout: float = 0.2,
        move_scalar_dim: int = 13,
        sq_embed_dim: int = 48,
        expert_hidden: int = 160,
        expert_layers: int = 2,
        mistake_expert_ver: str = 'default',
        time_expert_ver: str = 'default',
        wdl_expert_ver: str = 'default',
        move_head_ver: str = 'attention_deep',
        contrastive_embed_dim: int = 96,
        contrastive_hidden_dim: int = 192,
        contrastive_margin: float = 1.0,
        use_phase_experts: bool = True,
        phase_hidden_dim: int = 96,
        num_phases: int = 3,
        use_film: bool = True,
        film_hidden_dim: int = 96,
        use_tactical_enrichment: bool = False,
        tactical_preprocessor_path: Optional[str] = None,
        tactical_preprocessor_config: Optional[dict] = None,
    ):
        super().__init__()
        self.max_possible = max_possible
        self.hidden_dim = hidden_dim
        self.expert_hidden = expert_hidden
        self.expert_layers = expert_layers
        self.dropout = dropout
        self.contrastive_embed_dim = contrastive_embed_dim
        self.contrastive_margin = contrastive_margin
        self.use_phase_experts = use_phase_experts
        self.phase_hidden_dim = phase_hidden_dim
        self.num_phases = num_phases
        self.use_film = use_film
        self.use_tactical_enrichment = use_tactical_enrichment

        # ---- Encoders ----
        self.board_encoder = BoardEncoder(
            in_planes=23, channels=cnn_channels, num_res_blocks=num_res_blocks
        )
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_dim, output_dim=64, dropout=dropout * 0.5
        )

        # ---- FiLM conditioning (Elo-aware trunk modulation) ----
        if use_film:
            self.film_conditioner = FiLMConditioner(
                input_dim=tabular_dim,
                num_blocks=num_res_blocks,
                channels=cnn_channels,
                hidden_dim=film_hidden_dim,
            )
        else:
            self.film_conditioner = None

        # ---- Move encoder (replaces CNN-per-move) ----
        self.move_encoder = MoveEncoder(
            cnn_channels=cnn_channels,
            scalar_dim=move_scalar_dim,
            sq_embed_dim=sq_embed_dim,
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

        # ---- Expert Modules ----
        self._expert_versions = {
            'mistake': mistake_expert_ver,
            'time': time_expert_ver,
            'wdl': wdl_expert_ver,
        }
        self._move_head_version = move_head_ver

        if use_phase_experts:
            # Phase-gated experts: shared PhaseEncoder + 3 PhaseGatedExperts
            self.phase_encoder = PhaseEncoder(
                input_dim=hidden_dim, hidden_dim=phase_hidden_dim,
                num_phases=num_phases, dropout=dropout * 0.5,
            )
            self.mistake_expert = PhaseGatedExpert(
                hidden_dim, expert_hidden, 1,
                num_phases=num_phases, n_layers=expert_layers, dropout=dropout,
            )
            self.time_expert = PhaseGatedExpert(
                hidden_dim, expert_hidden, 1,
                num_phases=num_phases, n_layers=expert_layers, dropout=dropout,
            )
            self.wdl_before_expert = PhaseGatedExpert(
                hidden_dim, expert_hidden, 3,
                num_phases=num_phases, n_layers=expert_layers, dropout=dropout,
            )
            # Cross-feed dim uses expert_hidden directly (blended output shape)
            self._cross_feed_dim = hidden_dim + 3 * expert_hidden
        else:
            # V4-style single experts from registry
            self.phase_encoder = None
            self.mistake_expert = self._build_expert(mistake_expert_ver, 1)
            self.time_expert = self._build_expert(time_expert_ver, 1)
            self.wdl_before_expert = self._build_expert(wdl_expert_ver, 3)
            self._cross_feed_dim = hidden_dim + sum(
                _EXPERT_REGISTRY[v]['hidden_dim']
                for v in [mistake_expert_ver, time_expert_ver, wdl_expert_ver]
            )

        # ---- Contrastive Move Expert (V5) ----
        if contrastive_embed_dim > 0:
            self.contrastive_encoder = ContrastiveEncoder(
                input_dim=hidden_dim,  # move_emb dim
                hidden_dim=contrastive_hidden_dim,
                output_dim=contrastive_embed_dim,
                dropout=dropout,
            )
            self.contrastive_anchor_proj = nn.Linear(hidden_dim, contrastive_embed_dim)
        else:
            self.contrastive_encoder = None
            self.contrastive_anchor_proj = None

        # ---- Move Head (registry-backed, cross-feed from experts) ----
        # move_head_in = cross_feed_dim + move_emb(hidden_dim) + contrastive_embed_dim
        move_head_in = self._cross_feed_dim + hidden_dim + contrastive_embed_dim
        if use_phase_experts:
            self.move_head = PhaseGatedMoveHead(
                head_factory=lambda: _MOVE_HEAD_REGISTRY[move_head_ver]['factory'](
                    move_head_in, hidden_dim, dropout),
                num_phases=num_phases,
            )
        else:
            self.move_head = _MOVE_HEAD_REGISTRY[move_head_ver]['factory'](
                move_head_in, hidden_dim, dropout)

        # ---- Tactical Enrichment (optional, off by default) ----
        self.tactical_enrichment = None
        self.tactical_alpha = None
        self._tactical_config = None
        if use_tactical_enrichment:
            from tactical_models import TacticalPreprocessor
            if tactical_preprocessor_path is not None:
                preprocessor = TacticalPreprocessor.load(
                    tactical_preprocessor_path, device='cpu')
                self._tactical_config = preprocessor.config
            elif tactical_preprocessor_config is not None:
                preprocessor = TacticalPreprocessor(**tactical_preprocessor_config)
                self._tactical_config = tactical_preprocessor_config
            else:
                raise ValueError(
                    'use_tactical_enrichment=True requires either '
                    'tactical_preprocessor_path or tactical_preprocessor_config')
            preprocessor.freeze_for_v5()  # backbone+heads frozen, gate trainable
            tac_emb_dim = preprocessor.total_embedding_dim  # 384
            self.tactical_enrichment = nn.ModuleDict({
                'preprocessor': preprocessor,
                'projection': nn.Sequential(
                    nn.Linear(hidden_dim + tac_emb_dim, 384),
                    nn.Mish(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(384, hidden_dim),
                ),
            })
            # Residual alpha: sigmoid(-3.0) ~ 0.05, starts near pure trunk
            self.tactical_alpha = nn.Parameter(torch.tensor(-3.0))

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
    # Registry builders & hot-swap
    # ------------------------------------------------------------------

    def _build_expert(self, version: str, output_dim: int,
                      dropout: Optional[float] = None) -> nn.Module:
        """Instantiate an expert from the registry."""
        if version not in _EXPERT_REGISTRY:
            raise ValueError(f"Unknown expert version '{version}'. "
                             f"Available: {list(_EXPERT_REGISTRY.keys())}")
        d = dropout if dropout is not None else self.dropout
        return _EXPERT_REGISTRY[version]['factory'](self.hidden_dim, output_dim, d)

    def _build_move_head(self, version: str,
                         dropout: Optional[float] = None) -> nn.Module:
        """Instantiate a move head from the registry, phase-gated if enabled."""
        if version not in _MOVE_HEAD_REGISTRY:
            raise ValueError(f"Unknown move head version '{version}'. "
                             f"Available: {list(_MOVE_HEAD_REGISTRY.keys())}")
        d = dropout if dropout is not None else self.dropout
        move_head_in = self._cross_feed_dim + self.hidden_dim + self.contrastive_embed_dim
        if self.use_phase_experts:
            return PhaseGatedMoveHead(
                head_factory=lambda: _MOVE_HEAD_REGISTRY[version]['factory'](
                    move_head_in, self.hidden_dim, d),
                num_phases=self.num_phases,
            )
        return _MOVE_HEAD_REGISTRY[version]['factory'](move_head_in, self.hidden_dim, d)

    def _recompute_cross_feed_dim(self):
        """Update cross_feed_dim from current expert versions."""
        if self.use_phase_experts:
            self._cross_feed_dim = self.hidden_dim + 3 * self.expert_hidden
        else:
            self._cross_feed_dim = self.hidden_dim + sum(
                _EXPERT_REGISTRY[self._expert_versions[h]]['hidden_dim']
                for h in ['mistake', 'time', 'wdl']
            )

    def swap_expert(self, head_name: str, version: str,
                    dropout: Optional[float] = None,
                    rebuild_move_head: bool = True) -> None:
        """Hot-swap an expert module.  Rebuilds move head if hidden_dim changes.

        When use_phase_experts is True, swapping is not supported (all experts
        are PhaseGatedExpert and share phase weights).

        Args:
            head_name: 'mistake', 'time', or 'wdl'
            version: registered expert name
            dropout: dropout for new module (None = use model default)
            rebuild_move_head: auto-rebuild move head if cross_feed_dim changes
        """
        if self.use_phase_experts:
            raise RuntimeError(
                "Cannot hot-swap individual experts when use_phase_experts=True. "
                "All experts are PhaseGatedExpert and share the PhaseEncoder. "
                "Re-instantiate the model with use_phase_experts=False to use "
                "the expert registry swap mechanism."
            )

        output_dims = {'mistake': 1, 'time': 1, 'wdl': 3}
        attr_names = {
            'mistake': 'mistake_expert',
            'time': 'time_expert',
            'wdl': 'wdl_before_expert',
        }
        if head_name not in output_dims:
            raise ValueError(f"head_name must be one of {list(output_dims.keys())}")

        old_ver = self._expert_versions[head_name]
        old_hidden = _EXPERT_REGISTRY[old_ver]['hidden_dim']
        new_hidden = _EXPERT_REGISTRY[version]['hidden_dim']

        new_expert = self._build_expert(version, output_dims[head_name], dropout)
        setattr(self, attr_names[head_name], new_expert)
        self._expert_versions[head_name] = version
        self._recompute_cross_feed_dim()

        if old_hidden != new_hidden and rebuild_move_head:
            self.move_head = self._build_move_head(self._move_head_version, dropout)
            print(f"  [SWAP] Expert '{head_name}' hidden changed "
                  f"{old_hidden}→{new_hidden}, move head rebuilt (fresh weights)")

        n = sum(p.numel() for p in new_expert.parameters())
        print(f"  [SWAP] {head_name}_expert → '{version}' "
              f"(hidden={new_hidden}, params={n:,})")

    def swap_move_head(self, version: str,
                       dropout: Optional[float] = None) -> None:
        """Hot-swap the move head."""
        self.move_head = self._build_move_head(version, dropout)
        self._move_head_version = version
        n = sum(p.numel() for p in self.move_head.parameters())
        print(f"  [SWAP] move_head → '{version}' (params={n:,})")

    @property
    def component_versions(self) -> Dict[str, str]:
        """Return version strings for all swappable components."""
        versions = {
            **{f'{k}_expert': v for k, v in self._expert_versions.items()},
            'move_head': self._move_head_version,
        }
        if self.use_phase_experts:
            versions['phase_gating'] = 'enabled'
        if self.use_film:
            versions['film_conditioning'] = 'enabled'
        return versions

    def _cross_attend(self, context, move_emb, key_padding_mask):
        query = self.query_proj(context).unsqueeze(1)
        # Fix #1: a fully-padded row (zero legal moves) masks every key, making
        # softmax divide by zero → NaN that poisons the whole batch. Unmask the
        # first key for such rows; their outputs are ignored downstream (move_idx
        # = -1 → ignore_index in CE, and masked out of every other head).
        safe_mask = key_padding_mask.clone()
        all_padded = safe_mask.all(dim=1)
        # Branchless: unmask key 0 only on fully-padded rows (compile-safe).
        safe_mask[:, 0] = safe_mask[:, 0] & ~all_padded
        attn_out, _ = self.cross_attn(query, move_emb, move_emb, key_padding_mask=safe_mask)
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
        Full forward pass — single CNN + expert cross-feed + contrastive + phase gating.

        Returns
        -------
        dict with keys:
            move_logits, mistake_prob, win_prob_before, time_spent,
            contrastive_embed (if enabled), contrastive_anchor (if enabled),
            phase_weights (if use_phase_experts)
        """
        B = current_planes.shape[0]
        M = possible_from_sq.shape[1]

        # --- Sanitize inputs (fix #5 + #7: sentinel/non-finite → neutral) ---
        tabular = sanitize_features(tabular)
        possible_scalars = sanitize_features(possible_scalars)

        # Guard board planes against any non-finite leak (fix #8)
        current_planes = torch.nan_to_num(current_planes, nan=0.0, posinf=0.0, neginf=0.0)

        # --- FiLM conditioning (V5) ---
        film_params = None
        if self.film_conditioner is not None:
            film_params = self.film_conditioner(tabular)

        # --- Single CNN pass ---
        board_emb, feature_map = self.board_encoder(current_planes, film_params=film_params)

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

        # --- Tactical enrichment (optional, no-op when disabled) ---
        if self.tactical_enrichment is not None:
            gated_emb = self.tactical_enrichment['preprocessor'].get_gated_embedding(
                current_planes)  # (B, 384) — preprocessor is frozen
            enriched = torch.cat([trunk_out, gated_emb], dim=-1)
            projected = self.tactical_enrichment['projection'](enriched)
            alpha = torch.sigmoid(self.tactical_alpha)
            trunk_out = alpha * projected + (1.0 - alpha) * trunk_out

        outputs: Dict[str, torch.Tensor] = {}

        # ============================================================
        # PHASE DETECTION (V5, if enabled)
        # ============================================================
        phase_weights = None
        if self.use_phase_experts and self.phase_encoder is not None:
            phase_weights = self.phase_encoder(trunk_out)  # (B, num_phases)
            outputs['phase_weights'] = phase_weights

        # ============================================================
        # EXPERT HEADS (phase-gated or standard)
        # ============================================================
        mistake_hidden, mistake_logit = self.mistake_expert(trunk_out, phase_weights)
        outputs['mistake_prob'] = mistake_logit

        time_hidden, time_out = self.time_expert(trunk_out, phase_weights)
        outputs['time_spent'] = time_out

        wdl_before_hidden, wdl_before_logits = self.wdl_before_expert(trunk_out, phase_weights)
        outputs['win_prob_before'] = F.softmax(wdl_before_logits, dim=-1)
        # Fix #3: expose raw logits so the loss can use numerically-stable
        # log_softmax instead of log(softmax(...)). Consumers (infer/validate/
        # engine) continue to read the probability tensor above unchanged.
        outputs['win_prob_before_logits'] = wdl_before_logits

        # ============================================================
        # CONTRASTIVE ENCODER (V5, if enabled) — forced float32 for stability
        # ============================================================
        if self.contrastive_encoder is not None:
            with torch.amp.autocast('cuda', enabled=False):
                contrastive_embed = self.contrastive_encoder(move_emb.float())  # (B, M, 64)
                contrastive_embed = contrastive_embed * possible_mask.unsqueeze(-1)
                contrastive_anchor = self.contrastive_anchor_proj(trunk_out.float())  # (B, 64)

            outputs['contrastive_embed'] = contrastive_embed
            outputs['contrastive_anchor'] = contrastive_anchor

        # ============================================================
        # CROSS-FEED FUSION → Move Head
        # ============================================================
        cross_feed = torch.cat([
            trunk_out,
            wdl_before_hidden.detach(),
            mistake_hidden.detach(),
            time_hidden.detach(),
        ], dim=-1)  # (B, cross_feed_dim)

        cross_exp = cross_feed.unsqueeze(1).expand(-1, M, -1)

        # Build full move head input
        parts = [cross_exp, move_emb]
        if self.contrastive_encoder is not None:
            parts.append(contrastive_embed)
        else:
            # Zero-pad to keep move_head_in consistent (disabled contrastive)
            # Only needed if contrastive_embed_dim > 0 was expected but encoder is None
            pass
        full_input = torch.cat(parts, dim=-1)  # (B, M, move_head_in)

        # Dispatch to move head (phase-gated vs attention vs MLP)
        if isinstance(self.move_head, PhaseGatedMoveHead):
            move_scores = self.move_head(full_input, pad_mask, phase_weights)  # (B, M)
            move_scores = move_scores.to(torch.float32)
        elif isinstance(self.move_head, AttentionMoveHead):
            move_scores = self.move_head(full_input, pad_mask)  # (B, M)
            move_scores = move_scores.to(torch.float32)
        else:
            flat_scores = self.move_head(full_input.reshape(B * M, -1)).squeeze(-1)
            move_scores = flat_scores.reshape(B, M).to(torch.float32)

        # Fix #2: mask padded moves to -inf, but guard fully-padded rows.
        # A row with zero legal moves would become all -inf → softmax/CE NaN.
        # Such rows carry move_idx = -1 (ignored by CE) and are masked out of
        # every other head, so leaving their scores finite is behavior-neutral.
        safe_pad = pad_mask.clone()
        all_padded = safe_pad.all(dim=1)
        # Branchless: unmask col 0 only on fully-padded rows (compile-safe).
        safe_pad[:, 0] = safe_pad[:, 0] & ~all_padded
        move_scores = move_scores.masked_fill(safe_pad, float('-inf'))

        outputs['move_logits'] = move_scores

        return outputs


# ---------------------------------------------------------------------------
# Loss V5 (adds contrastive triplet loss to V4's Kendall-weighted loss)
# ---------------------------------------------------------------------------

class MIMOLossV5(nn.Module):
    """
    Multi-task loss with Kendall uncertainty weighting + contrastive triplet loss.

    Same as V4's MIMOLoss for the original 4 heads, plus a 5th 'contrastive'
    head with its own Kendall log_var.  The contrastive loss uses self-referential
    near-miss selection: the model's own highest-scored non-chosen move.

    When contrastive outputs are absent (contrastive disabled), the 'contrastive'
    log_var is unused and the loss falls back to 4-head behavior.
    """

    HEADS = ['move_logits', 'mistake_prob', 'win_prob_before', 'time_spent', 'contrastive']

    def __init__(self, contrastive_margin: float = 1.0):
        super().__init__()
        self.contrastive_margin = contrastive_margin
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
            # Fix #3: prefer numerically-stable log_softmax over the raw logits;
            # fall back to log(prob) only if logits weren't provided.
            if 'win_prob_before_logits' in predictions:
                log_pred = F.log_softmax(predictions['win_prob_before_logits'], dim=-1)
            else:
                log_pred = torch.log(predictions['win_prob_before'].clamp(min=1e-8))
            losses['win_prob_before'] = -(targets['win_prob_before'] * log_pred).sum(-1).mean()

        if 'time_spent' in predictions and 'time_spent_log' in targets:
            losses['time_spent'] = F.huber_loss(
                predictions['time_spent'].squeeze(-1),
                targets['time_spent_log'], delta=2.0,
            )

        # ============================================================
        # SOFT CONTRASTIVE TRIPLET LOSS (V5)
        #
        # Confidence-weighted: when the model's top-3 moves have similar
        # probabilities (ambiguous position), the loss weight → 0.  This
        # prevents the triplet loss from fighting itself on true 50/50
        # positions where multiple moves are equally human-playable.
        #
        # Confidence = 1 - entropy(top3_probs) / log(3)
        #   - Decisive (one dominant move):  confidence → 1, full loss
        #   - Ambiguous (uniform top-3):     confidence → 0, ~zero loss
        # ============================================================
        if ('contrastive_embed' in predictions and
                'contrastive_anchor' in predictions and
                'move_idx' in targets):
            anchor = predictions['contrastive_anchor']      # (B, 64)
            c_embed = predictions['contrastive_embed']      # (B, M, 64)
            move_logits = predictions['move_logits']        # (B, M)
            actual_idx = targets['move_idx']                # (B,)

            B = anchor.shape[0]

            # Only compute for valid examples (actual_idx >= 0)
            valid = actual_idx >= 0
            # Fix #8: skip forced/degenerate positions (<2 legal moves). Their
            # near-miss negative is selected by masking the true move then argmax,
            # which lands on a padded (zeroed) embedding → a meaningless triplet.
            # Padded moves are already -inf in move_logits, so finite count = legal count.
            legal_count = torch.isfinite(move_logits).sum(dim=1)   # (B,)
            valid = valid & (legal_count >= 2)
            if valid.any():
                v_anchor = anchor[valid]                    # (V, 64)
                v_embed = c_embed[valid]                    # (V, M, 64)
                v_logits = move_logits[valid]               # (V, M)
                v_actual = actual_idx[valid]                # (V,)
                V = v_anchor.shape[0]

                # --- Confidence weighting from top-3 probability spread ---
                # Detached: confidence weights should not produce gradients
                with torch.no_grad():
                    probs = F.softmax(v_logits, dim=-1)             # (V, M)
                    top3_probs, _ = probs.topk(3, dim=-1)           # (V, 3)
                    # Normalize top-3 to a distribution
                    top3_norm = top3_probs / top3_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    # Entropy of top-3 distribution, max = log(3) for uniform
                    top3_entropy = -(top3_norm * torch.log(top3_norm.clamp(min=1e-8))).sum(dim=-1)
                    # Confidence: 0 = uniform top-3 (ambiguous), 1 = dominant move (decisive)
                    confidence = 1.0 - top3_entropy / math.log(3)   # (V,)
                    confidence = confidence.clamp(min=0.0, max=1.0)

                # Positive: embedding of the human's chosen move
                positive = v_embed[torch.arange(V, device=v_embed.device), v_actual]  # (V, 64)

                # Self-referential near-miss: highest model score that isn't ground truth
                # .detach() on logits so gradients don't flow through argmax selection
                logits_for_nearmiss = v_logits.detach().clone()
                logits_for_nearmiss[torch.arange(V, device=v_logits.device), v_actual] = float('-inf')
                near_miss_idx = logits_for_nearmiss.argmax(dim=1)  # (V,)
                negative = v_embed[torch.arange(V, device=v_embed.device), near_miss_idx]  # (V, 64)

                # Triplet margin loss, weighted by confidence
                d_pos = F.pairwise_distance(v_anchor, positive, eps=1e-6)
                d_neg = F.pairwise_distance(v_anchor, negative, eps=1e-6)
                per_sample_loss = F.relu(d_pos - d_neg + self.contrastive_margin)  # (V,)
                # Weight by confidence: ambiguous positions contribute ~0
                weighted_loss = per_sample_loss * confidence
                contrastive_loss = weighted_loss.sum() / confidence.sum().clamp(min=1.0)

                losses['contrastive'] = contrastive_loss

        # ============================================================
        # KENDALL MULTI-TASK WEIGHTING (now with up to 5 heads)
        # ============================================================
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for name, loss_val in losses.items():
            log_var = self.log_vars[name].clamp(-4.0, 4.0)
            precision = torch.exp(-log_var)
            total = total + precision * loss_val + log_var

        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict['total'] = total.item()
        for name in losses:
            loss_dict[f'w_{name}'] = torch.exp(-self.log_vars[name].clamp(-4.0, 4.0)).item()
        return total, loss_dict


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(42)
    B, M = 4, 40

    print("=" * 60)
    print("  V5 FULL (contrastive + phase-gated + default MLP head)")
    print("=" * 60)
    model = ChessMIMOModelV5(max_possible=M, use_phase_experts=True,
                              contrastive_embed_dim=64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"V5 Model parameters: {n_params:,}")
    print(f"Component versions: {model.component_versions}")
    print(f"Contrastive embed dim: {model.contrastive_embed_dim}")
    print(f"Contrastive margin: {model.contrastive_margin}")
    print(f"Phase experts: {model.use_phase_experts}")
    print(f"FiLM conditioning: {model.use_film}")

    current = torch.randn(B, 23, 8, 8)
    from_sq = torch.randint(0, 64, (B, M))
    to_sq = torch.randint(0, 64, (B, M))
    promo = torch.zeros(B, M, dtype=torch.long)
    poss_scalars = torch.randn(B, M, 13)
    poss_mask = torch.ones(B, M)
    poss_mask[:, 25:] = 0
    tabular = torch.randn(B, 20)
    actual_idx = torch.randint(0, 25, (B,))

    outputs = model(current, from_sq, to_sq, promo, poss_scalars, poss_mask, tabular, actual_idx)
    print("\nOutputs:")
    for k, v in outputs.items():
        print(f"  {k:20s} {tuple(v.shape)}")

    targets = {
        'move_idx': torch.randint(0, 25, (B,)),
        'is_mistake': torch.randint(0, 2, (B,)).float(),
        'win_prob_before': F.softmax(torch.randn(B, 3), dim=-1),
        'time_spent_log': torch.rand(B) * 4,
    }
    criterion = MIMOLossV5(contrastive_margin=1.0)
    loss, ld = criterion(outputs, targets)
    print(f"\nTotal loss: {loss.item():.4f}")
    for k, v in ld.items():
        print(f"  {k:25s} {v:.4f}")

    # Phase weight distribution
    if 'phase_weights' in outputs:
        pw = outputs['phase_weights'].detach().mean(dim=0)
        print(f"\n  Phase weights (avg): opening={pw[0]:.3f}  midgame={pw[1]:.3f}  endgame={pw[2]:.3f}")

    # ----- Attention move head -----
    print("\n" + "=" * 60)
    print("  V5 with ATTENTION move head")
    print("=" * 60)
    model_attn = ChessMIMOModelV5(max_possible=M, use_phase_experts=True,
                                   contrastive_embed_dim=64, move_head_ver='attention')
    n_params_attn = sum(p.numel() for p in model_attn.parameters())
    print(f"V5 (attention) params: {n_params_attn:,}")
    out_attn = model_attn(current, from_sq, to_sq, promo, poss_scalars, poss_mask, tabular, actual_idx)
    print(f"  move_logits shape: {tuple(out_attn['move_logits'].shape)}")
    loss_attn, _ = criterion(out_attn, targets)
    print(f"  loss: {loss_attn.item():.4f}")

    # ----- V5 with phase experts disabled (V4 + contrastive only) -----
    print("\n" + "=" * 60)
    print("  V5 NO PHASE (contrastive only, V4-style experts)")
    print("=" * 60)
    model_nophase = ChessMIMOModelV5(max_possible=M, use_phase_experts=False,
                                      contrastive_embed_dim=64)
    n_p2 = sum(p.numel() for p in model_nophase.parameters())
    print(f"V5 (no phase) params: {n_p2:,}")
    out2 = model_nophase(current, from_sq, to_sq, promo, poss_scalars, poss_mask, tabular, actual_idx)
    print("Outputs:")
    for k, v in out2.items():
        print(f"  {k:20s} {tuple(v.shape)}")
    assert 'phase_weights' not in out2, "phase_weights should not be in output when disabled"
    print("  ✓ No phase_weights in output")

    # ----- V5 with contrastive disabled -----
    print("\n" + "=" * 60)
    print("  V5 NO CONTRASTIVE (phase experts only)")
    print("=" * 60)
    model_nocontr = ChessMIMOModelV5(max_possible=M, use_phase_experts=True,
                                      contrastive_embed_dim=0)
    n_p3 = sum(p.numel() for p in model_nocontr.parameters())
    print(f"V5 (no contrastive) params: {n_p3:,}")
    out3 = model_nocontr(current, from_sq, to_sq, promo, poss_scalars, poss_mask, tabular, actual_idx)
    assert 'contrastive_embed' not in out3, "contrastive_embed should not be in output"
    assert 'phase_weights' in out3, "phase_weights should be in output"
    print("  ✓ No contrastive outputs, phase_weights present")

    # ----- Inference mode -----
    print("\n--- Inference mode ---")
    outputs_infer = model(current, from_sq, to_sq, promo, poss_scalars, poss_mask, tabular)
    for k, v in outputs_infer.items():
        print(f"  {k:20s} {tuple(v.shape)}")

    # ----- V4 → V5 checkpoint compatibility -----
    print("\n--- V4→V5 checkpoint compatibility ---")
    model_v5 = ChessMIMOModelV5(max_possible=M, use_phase_experts=True)
    v5_keys = set(model_v5.state_dict().keys())
    # Simulate V4 state: only keys that exist in both V4 and V5 (no phase/contrastive)
    new_v5_prefixes = ('contrastive_encoder.', 'contrastive_anchor_proj.',
                       'phase_encoder.', 'mistake_expert.phase_experts.',
                       'time_expert.phase_experts.', 'wdl_before_expert.phase_experts.')
    v4_state = {k: v for k, v in model.state_dict().items()
                if not any(k.startswith(p) for p in new_v5_prefixes)}
    missing, unexpected = model_v5.load_state_dict(v4_state, strict=False)
    print(f"  Missing keys (expected — new V5 modules): {len(missing)}")
    for k in sorted(missing)[:10]:
        print(f"    {k}")
    if len(missing) > 10:
        print(f"    ... and {len(missing) - 10} more")
    print(f"  Unexpected keys: {len(unexpected)}")
