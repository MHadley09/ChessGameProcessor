#!/usr/bin/env python3
"""
Inference — Single-position prediction using TacticalPreprocessor
==================================================================
Loads a trained TacticalPreprocessor checkpoint and runs inference
on a single FEN position, printing theme predictions and opening
predictions from the shared backbone.

Usage:
    python infer_theme.py \
        --model checkpoints/tactical/best.pt \
        --fen "r1bqkbnr/pppppppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
"""

import argparse
import json

import chess
import torch

from tactical_models import TacticalPreprocessor
from puzzle_dataset_processor import board_to_planes


# 62 Lichess puzzle theme names (canonical order)
THEME_NAMES = [
    "advancedPawn", "advantage", "anapitstkyMate", "arabianMate",
    "attackingF2F7", "attraction", "backRankMate", "bishopEndgame",
    "bodenMate", "capturingDefender", "castling", "clearance",
    "crushing", "defensiveMove", "deflection", "discoveredAttack",
    "doubleBishopMate", "doubleCheck", "endgame", "enPassant",
    "equalityVariation", "exposedKing", "fork", "hangingPiece",
    "hookMate", "interference", "intermezzo", "kingsideAttack",
    "knightEndgame", "long", "master", "masterVsMaster",
    "mate", "mateIn1", "mateIn2", "mateIn3",
    "mateIn4", "mateIn5", "middlegame", "oneMove",
    "opening", "pawnEndgame", "pin", "promotion",
    "queenEndgame", "queenRookEndgame", "queensideAttack", "quietMove",
    "rookEndgame", "sacrifice", "short", "skewer",
    "smotheredMate", "superGM", "trappedPiece", "underPromotion",
    "veryLong", "xRayAttack", "zugzwang",
    # Padding to 62 — actual count may differ; trim or extend as needed
    "reservedTheme1", "reservedTheme2", "reservedTheme3",
]


def infer(model_path: str, fen: str, device: str = "cpu", threshold: float = 0.3):
    """Run inference on a single position."""

    # Load model
    model = TacticalPreprocessor.load(model_path, device=device)
    model.eval()

    # Encode position
    board = chess.Board(fen)
    planes = board_to_planes(board)
    planes_t = torch.tensor(planes, dtype=torch.float32).unsqueeze(0).to(device)

    # Forward pass (one backbone pass, both heads)
    with torch.no_grad():
        out = model(planes_t)

    # Theme predictions
    theme_probs = torch.sigmoid(out["theme_logits"][0]).cpu().numpy()
    confidence = out["confidence"][0, 0].cpu().item()

    print(f"Position: {fen}")
    print(f"Side to move: {'White' if board.turn else 'Black'}")
    print(f"Gate confidence: {confidence:.4f}")
    print()

    # Themes above threshold
    print(f"Themes (threshold={threshold}):")
    active_themes = []
    for i, prob in enumerate(theme_probs):
        if prob >= threshold:
            name = THEME_NAMES[i] if i < len(THEME_NAMES) else f"theme_{i}"
            active_themes.append((name, prob))

    if active_themes:
        active_themes.sort(key=lambda x: -x[1])
        for name, prob in active_themes:
            print(f"  {name:25s} {prob:.4f}")
    else:
        print("  (none above threshold)")

    # Opening predictions (top 5)
    print()
    print("Opening predictions (top 5):")
    opening_probs = torch.softmax(out["opening_logits"][0], dim=-1).cpu()
    topk_probs, topk_idxs = opening_probs.topk(5)
    for rank, (prob, idx) in enumerate(zip(topk_probs, topk_idxs), 1):
        print(f"  {rank}. class_{idx.item():4d}  {prob.item():.4f}")

    # Embedding norms
    print()
    tactical_norm = out["tactical_embedding"][0].norm().item()
    opening_norm = out["opening_embedding"][0].norm().item()
    gated_norm = out["gated_embedding"][0].norm().item()
    print(f"Embedding norms:")
    print(f"  Tactical:  {tactical_norm:.4f}")
    print(f"  Opening:   {opening_norm:.4f}")
    print(f"  Gated:     {gated_norm:.4f}")

    return out


def main():
    parser = argparse.ArgumentParser(description="Inference with TacticalPreprocessor")
    parser.add_argument("--model", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--fen", type=str,
        default="r1bqkbnr/pppppppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        help="FEN string",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    infer(args.model, args.fen, device=args.device, threshold=args.threshold)


if __name__ == "__main__":
    main()
