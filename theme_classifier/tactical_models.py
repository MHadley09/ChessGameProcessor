#!/usr/bin/env python3
"""
Tactical Models — Shared Backbone with Theme + Opening Expert Heads
====================================================================
Single CNN backbone produces both tactical (theme) and opening embeddings
in one forward pass. Two expert heads branch off the shared trunk.

Architecture:
    board_planes (B, 23, 8, 8)
        → SharedBackbone (ResNet-SE blocks)
        → global avg pool → trunk_features (B, channels)
        ├→ ThemeHead → tactical_embedding (B, 256) → theme_logits (B, 62)
        └→ OpeningHead → opening_embedding (B, 128) → opening_logits (B, ~350)

The combined 384-dim embedding (with confidence gate) is injected into V5
via trunk enrichment (Option A).

Training: Joint multi-task loss (BCE for themes + CE for openings) in
train_classifier.py. Both heads train simultaneously on puzzle data.

V5 Integration: Entire model is frozen. Confidence gate + projection
layer are the only trainable components during V5 fine-tuning.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-Excitation block."""

    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        mid = max(channels // ratio, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        se = x.mean(dim=(2, 3))           # (B, C)
        se = F.mish(self.fc1(se))          # (B, mid)
        se = torch.sigmoid(self.fc2(se))   # (B, C)
        return x * se.unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    """Residual block: Conv → BN → Mish → Conv → BN → SE → residual → Mish."""

    def __init__(self, channels: int, se_ratio: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.mish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.mish(out + residual)


# ---------------------------------------------------------------------------
# Shared CNN Backbone
# ---------------------------------------------------------------------------

class ChessCNNBackbone(nn.Module):
    """Shared CNN backbone for board feature extraction.
    
    Architecture:
      Input:  (B, in_planes, 8, 8)
      Stem:   Conv2d in_planes → channels (3×3) + BN + Mish
      Body:   N × ResBlock with SE
      Output: Global average pool → (B, channels)
    """

    def __init__(
        self,
        in_planes: int = 23,
        channels: int = 128,
        num_blocks: int = 8,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Mish(),
        )
        self.blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_blocks)]
        )
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_planes, 8, 8) → (B, channels)"""
        x = self.stem(x)
        x = self.blocks(x)
        return x.mean(dim=(2, 3))  # Global average pool


# ---------------------------------------------------------------------------
# Expert Heads
# ---------------------------------------------------------------------------

class ThemeHead(nn.Module):
    """Multi-label theme expert: trunk → tactical_embedding → theme_logits.
    
    Loss: BCEWithLogitsLoss (multi-label, 62 themes).
    """

    def __init__(
        self,
        trunk_dim: int = 128,
        embedding_dim: int = 256,
        n_themes: int = 62,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_themes = n_themes

        self.projection = nn.Sequential(
            nn.Linear(trunk_dim, embedding_dim * 2),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, n_themes)

    def forward(
        self, trunk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            trunk: (B, trunk_dim) from shared backbone
            
        Returns:
            logits: (B, n_themes) — raw logits, apply sigmoid for probs
            embedding: (B, embedding_dim) — tactical embedding for V5
        """
        embedding = self.projection(trunk)
        logits = self.classifier(embedding)
        return logits, embedding


class OpeningHead(nn.Module):
    """Multi-class opening expert: trunk → opening_embedding → opening_logits.
    
    Loss: CrossEntropyLoss (multi-class, ~350 openings).
    Only trained on positions with opening labels (masked loss for positions
    past move 20 or without OpeningTags).
    """

    def __init__(
        self,
        trunk_dim: int = 128,
        embedding_dim: int = 128,
        n_openings: int = 350,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_openings = n_openings

        self.projection = nn.Sequential(
            nn.Linear(trunk_dim, embedding_dim * 2),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, n_openings)

    def forward(
        self, trunk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            trunk: (B, trunk_dim) from shared backbone
            
        Returns:
            logits: (B, n_openings) — raw logits, apply softmax for probs
            embedding: (B, embedding_dim) — opening embedding for V5
        """
        embedding = self.projection(trunk)
        logits = self.classifier(embedding)
        return logits, embedding


# ---------------------------------------------------------------------------
# TacticalPreprocessor — The Primary Model
# ---------------------------------------------------------------------------

class TacticalPreprocessor(nn.Module):
    """Shared-backbone tactical + opening classifier with confidence gate.
    
    This is the primary model for:
      1. Training (joint multi-task loss on puzzle data)
      2. V5 integration (frozen backbone + heads, trainable gate)
    
    Architecture:
        board_planes (B, 23, 8, 8)
            → ChessCNNBackbone → trunk (B, backbone_channels)
            ├→ ThemeHead → tactical_emb (B, 256), theme_logits (B, 62)
            └→ OpeningHead → opening_emb (B, 128), opening_logits (B, ~350)
            
            combined_emb = cat(tactical_emb, opening_emb)  # (B, 384)
            confidence = gate(combined_emb)                  # (B, 1) sigmoid
            gated_emb = combined_emb * confidence            # (B, 384)
    
    Training modes:
        - Full training (puzzle data): all parameters trainable
        - V5 fine-tuning: backbone + heads frozen, only gate trainable
        - Pure inference: everything frozen
    """

    def __init__(
        self,
        in_planes: int = 23,
        backbone_channels: int = 128,
        backbone_blocks: int = 8,
        n_themes: int = 62,
        n_openings: int = 350,
        theme_embedding_dim: int = 256,
        opening_embedding_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Shared backbone — one CNN, one forward pass
        self.backbone = ChessCNNBackbone(
            in_planes=in_planes,
            channels=backbone_channels,
            num_blocks=backbone_blocks,
        )

        # Expert heads branch off the shared trunk
        self.theme_head = ThemeHead(
            trunk_dim=backbone_channels,
            embedding_dim=theme_embedding_dim,
            n_themes=n_themes,
            dropout=dropout,
        )
        self.opening_head = OpeningHead(
            trunk_dim=backbone_channels,
            embedding_dim=opening_embedding_dim,
            n_openings=n_openings,
            dropout=dropout,
        )

        # Dimensions
        self.theme_embedding_dim = theme_embedding_dim
        self.opening_embedding_dim = opening_embedding_dim
        self.total_embedding_dim = theme_embedding_dim + opening_embedding_dim

        # Confidence gate: learns when to trust the combined embedding
        # Stays trainable during V5 fine-tuning even when backbone+heads frozen
        self.confidence_gate = nn.Sequential(
            nn.Linear(self.total_embedding_dim, 64),
            nn.Mish(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Store config for serialization
        self.config = {
            "in_planes": in_planes,
            "backbone_channels": backbone_channels,
            "backbone_blocks": backbone_blocks,
            "n_themes": n_themes,
            "n_openings": n_openings,
            "theme_embedding_dim": theme_embedding_dim,
            "opening_embedding_dim": opening_embedding_dim,
            "dropout": dropout,
        }

    def forward(
        self, board_planes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass — one backbone pass, both expert heads.
        
        Args:
            board_planes: (B, 23, 8, 8)
            
        Returns:
            dict with:
              - theme_logits: (B, n_themes)
              - tactical_embedding: (B, theme_embedding_dim)
              - opening_logits: (B, n_openings)
              - opening_embedding: (B, opening_embedding_dim)
              - combined_embedding: (B, total_embedding_dim)
              - confidence: (B, 1)
              - gated_embedding: (B, total_embedding_dim)
        """
        # Single backbone forward pass
        trunk = self.backbone(board_planes)  # (B, backbone_channels)

        # Two expert heads, same trunk
        theme_logits, tactical_emb = self.theme_head(trunk)
        opening_logits, opening_emb = self.opening_head(trunk)

        # Combined embedding + confidence gate
        combined = torch.cat([tactical_emb, opening_emb], dim=-1)
        confidence = self.confidence_gate(combined)

        return {
            "theme_logits": theme_logits,
            "tactical_embedding": tactical_emb,
            "opening_logits": opening_logits,
            "opening_embedding": opening_emb,
            "combined_embedding": combined,
            "confidence": confidence,
            "gated_embedding": combined * confidence,
        }

    def get_gated_embedding(
        self, board_planes: torch.Tensor
    ) -> torch.Tensor:
        """Get confidence-gated combined embedding (for V5 integration).
        
        One backbone pass → two expert projections → gate → scaled embedding.
        This is the method V5 calls during its forward pass.
        
        Args:
            board_planes: (B, 23, 8, 8)
            
        Returns:
            (B, total_embedding_dim) gated embedding
        """
        trunk = self.backbone(board_planes)
        _, tactical_emb = self.theme_head(trunk)
        _, opening_emb = self.opening_head(trunk)
        combined = torch.cat([tactical_emb, opening_emb], dim=-1)
        confidence = self.confidence_gate(combined)
        return combined * confidence

    def get_combined_embedding(
        self, board_planes: torch.Tensor
    ) -> torch.Tensor:
        """Get concatenated embedding without gating (ungated).
        
        Args:
            board_planes: (B, 23, 8, 8)
            
        Returns:
            (B, total_embedding_dim) combined embedding
        """
        trunk = self.backbone(board_planes)
        _, tactical_emb = self.theme_head(trunk)
        _, opening_emb = self.opening_head(trunk)
        return torch.cat([tactical_emb, opening_emb], dim=-1)

    # -------------------------------------------------------------------
    # Freeze controls
    # -------------------------------------------------------------------

    def freeze_for_v5(self):
        """Freeze backbone + expert heads. Only confidence gate trains.
        
        Use this during V5 fine-tuning. The entire classifier is a
        frozen feature extractor; the gate learns when to suppress.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.theme_head.parameters():
            param.requires_grad = False
        for param in self.opening_head.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.theme_head.eval()
        self.opening_head.eval()
        # confidence_gate stays trainable

    def freeze(self):
        """Freeze everything including the gate (pure inference)."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    # -------------------------------------------------------------------
    # Prediction helpers
    # -------------------------------------------------------------------

    def predict_themes(
        self,
        board_planes: torch.Tensor,
        threshold: float = 0.5,
        theme_names: Optional[list] = None,
    ) -> list:
        """Predict theme labels for a batch of positions."""
        with torch.no_grad():
            out = self.forward(board_planes)
            probs = torch.sigmoid(out["theme_logits"])

        results = []
        for i in range(probs.size(0)):
            active = (probs[i] >= threshold).nonzero(as_tuple=True)[0].tolist()
            if theme_names:
                results.append([theme_names[j] for j in active])
            else:
                results.append(active)
        return results

    def predict_opening(
        self,
        board_planes: torch.Tensor,
        top_k: int = 5,
        opening_names: Optional[dict] = None,
    ) -> list:
        """Predict top-k opening labels for a batch of positions."""
        with torch.no_grad():
            out = self.forward(board_planes)
            probs = F.softmax(out["opening_logits"], dim=-1)

        results = []
        topk_probs, topk_idxs = probs.topk(top_k, dim=-1)
        for i in range(probs.size(0)):
            preds = []
            for j in range(top_k):
                idx = topk_idxs[i, j].item()
                p = topk_probs[i, j].item()
                name = (
                    opening_names.get(idx, f"opening_{idx}")
                    if opening_names
                    else idx
                )
                preds.append({"label": name, "probability": p, "rank": j + 1})
            results.append(preds)
        return results

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def save(self, path: str):
        """Save full model state + config."""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(
        cls, path: str, device: str = "cpu"
    ) -> "TacticalPreprocessor":
        """Load from checkpoint."""
        data = torch.load(path, map_location=device, weights_only=False)
        model = cls(**data["config"])
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model
