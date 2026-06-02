"""
Eval Accuracy Benchmark
=======================
Compares LC0 evaluation accuracy across configs using
XL with 1s search as ground truth.

Configs:
  large_1node  — large weights, 1 node per move (current pipeline)
  large_1ms    — large weights, 1ms per position
  large_5ms    — large weights, 5ms per position
  large_25ms   — large weights, 25ms per position
  large_50ms   — large weights, 50ms per position
  xl_25ms      — XL weights, 25ms per position
  xl_1s_GT     — XL weights, 1s per position (ground truth)

Usage:
  python eval_accuracy_benchmark.py <pgn_file>
      --lc0 <path_to_lc0>
      --large-weights <path_to_large.pb.gz>
      --xl-weights <path_to_xl.pb.gz>
      [--num-positions 1000]
      [--backend cuda-fp16]
      [--batch-size 128]
"""

import argparse
import random
import sys
import time
from typing import List, Dict, Optional, Tuple

import chess
import chess.pgn
import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_evaluator import SyncBatchEvaluator
from direct_evaluator import SyncDirectEvaluator


def pick_middlegame_positions(pgn_path: str, num_positions: int = 1000,
                               min_ply: int = 16, max_ply: int = 60,
                               seed: int = 42) -> List[Tuple[str, str]]:
    """Pick random middlegame positions from different games."""
    rng = random.Random(seed)
    candidates = []

    print(f"[SCAN] Scanning PGN for middlegame positions (ply {min_ply}-{max_ply})...")
    games_scanned = 0
    max_scan = max(num_positions * 5, 10000)

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        while games_scanned < max_scan:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games_scanned += 1

            board = game.board()
            node = game
            ply = 0
            game_positions = []

            while node.variations:
                next_node = node.variation(0)
                board.push(next_node.move)
                ply += 1

                if min_ply <= ply <= max_ply:
                    num_pieces = len(board.piece_map())
                    if num_pieces >= 12 and not board.is_game_over():
                        num_legal = len(list(board.legal_moves))
                        if num_legal >= 5:
                            w = game.headers.get("White", "?")
                            b = game.headers.get("Black", "?")
                            info = f"{w} vs {b} (ply {ply})"
                            game_positions.append((board.fen(), info))

                node = next_node

            if game_positions:
                candidates.append(rng.choice(game_positions))

    print(f"[SCAN] Scanned {games_scanned} games, found {len(candidates)} candidates")

    if len(candidates) < num_positions:
        print(f"[WARN] Only found {len(candidates)} positions, wanted {num_positions}")
        return candidates

    return rng.sample(candidates, num_positions)


def _compute_ev(e: Dict, stm_is_white: bool) -> Optional[float]:
    """EV = win prob from STM perspective = stm_W + 0.5*D."""
    w = e.get("wdl_w")
    d = e.get("wdl_d")
    l = e.get("wdl_l")
    if w is None or d is None or l is None:
        return None
    if stm_is_white:
        return w + 0.5 * d
    else:
        return l + 0.5 * d


def evaluate_position_config(engine, board: chess.Board) -> List[Dict]:
    """Evaluate all legal moves, return sorted by EV descending."""
    results = engine.evaluate_all_legal_moves(board)
    stm_is_white = board.turn == chess.WHITE

    for e in results:
        e["ev"] = _compute_ev(e, stm_is_white)

    def sort_key(e):
        if e["ev"] is not None:
            return e["ev"]
        if e.get("score_mate") is not None:
            m = e["score_mate"]
            return 100000 - m if m > 0 else -100000 - m
        return 0.0

    results.sort(key=sort_key, reverse=True)
    return results


def compare_configs(positions, configs, ground_truth_key):
    """Run all configs on all positions and compute deltas vs ground truth."""

    all_results = {}

    for name, engine in configs.items():
        print(f"\n[EVAL] Running config: {name}")
        all_results[name] = {}
        t0 = time.time()

        for idx, (fen, info) in enumerate(positions):
            board = chess.Board(fen)
            try:
                evals = evaluate_position_config(engine, board)
                all_results[name][idx] = evals

                # One debug line for first position
                if idx == 0 and evals:
                    inner = engine._evaluator
                    n_legal = len(list(board.legal_moves))
                    tpm = getattr(inner, 'time_per_move_ms', None)
                    top_nodes = evals[0].get('nodes', '?')
                    if tpm is not None:
                        print(f"  [DEBUG] n_legal={n_legal}, time_ms={tpm}, "
                              f"top_move_nodes={top_nodes}, moves_returned={len(evals)}")
                    else:
                        nm = getattr(inner, 'nodes_mult', None)
                        print(f"  [DEBUG] n_legal={n_legal}, nodes_mult={nm}, "
                              f"top_move_nodes={top_nodes}, moves_returned={len(evals)}")

            except Exception as e:
                print(f"  [ERR] Position {idx}: {e}")
                all_results[name][idx] = []

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  ... {idx+1}/{len(positions)} ({elapsed:.0f}s)")

        elapsed = time.time() - t0
        print(f"  Completed {len(positions)} positions in {elapsed:.1f}s "
              f"({elapsed/len(positions):.3f}s/pos)")

    # ── Compute deltas vs ground truth ──────────────────────────────────
    gt = all_results[ground_truth_key]
    comparison_configs = [k for k in configs if k != ground_truth_key]

    stats = {}
    for name in comparison_configs:
        ev_deltas_all = []
        ev_deltas_top5 = []
        ev_deltas_best = []
        rank_deltas = []
        top1_matches = 0
        top3_matches = 0
        positions_compared = 0

        for idx in range(len(positions)):
            gt_evals = gt.get(idx, [])
            cfg_evals = all_results[name].get(idx, [])
            if not gt_evals or not cfg_evals:
                continue
            positions_compared += 1

            cfg_by_move = {}
            for rank, e in enumerate(cfg_evals):
                cfg_by_move[e["move_uci"]] = (rank, e)

            if gt_evals[0]["move_uci"] == cfg_evals[0]["move_uci"]:
                top1_matches += 1

            cfg_top3_moves = {e["move_uci"] for e in cfg_evals[:3]}
            if gt_evals[0]["move_uci"] in cfg_top3_moves:
                top3_matches += 1

            gt_best_ev = gt_evals[0].get("ev")
            cfg_best_ev = cfg_evals[0].get("ev")
            if gt_best_ev is not None and cfg_best_ev is not None:
                ev_deltas_best.append(abs(gt_best_ev - cfg_best_ev))

            for gt_rank, gt_eval in enumerate(gt_evals):
                uci = gt_eval["move_uci"]
                if uci not in cfg_by_move:
                    continue
                cfg_rank, cfg_eval = cfg_by_move[uci]

                gt_ev = gt_eval.get("ev")
                cfg_ev = cfg_eval.get("ev")
                if gt_ev is not None and cfg_ev is not None:
                    delta = abs(gt_ev - cfg_ev)
                    ev_deltas_all.append(delta)
                    if gt_rank < 5:
                        ev_deltas_top5.append(delta)

                rank_deltas.append(abs(gt_rank - cfg_rank))

        n_pos = positions_compared
        stats[name] = {
            "ev_mean_all": np.mean(ev_deltas_all) if ev_deltas_all else 0,
            "ev_med_all": np.median(ev_deltas_all) if ev_deltas_all else 0,
            "ev_p95_all": np.percentile(ev_deltas_all, 95) if ev_deltas_all else 0,
            "ev_max_all": max(ev_deltas_all) if ev_deltas_all else 0,
            "ev_mean_top5": np.mean(ev_deltas_top5) if ev_deltas_top5 else 0,
            "ev_med_top5": np.median(ev_deltas_top5) if ev_deltas_top5 else 0,
            "ev_mean_best": np.mean(ev_deltas_best) if ev_deltas_best else 0,
            "ev_med_best": np.median(ev_deltas_best) if ev_deltas_best else 0,
            "ev_p95_best": np.percentile(ev_deltas_best, 95) if ev_deltas_best else 0,
            "mean_rank_delta": np.mean(rank_deltas) if rank_deltas else 0,
            "top1_accuracy": top1_matches / n_pos * 100 if n_pos else 0,
            "top3_accuracy": top3_matches / n_pos * 100 if n_pos else 0,
            "n_moves_compared": len(ev_deltas_all),
            "n_positions": n_pos,
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description="LC0 Eval Accuracy Benchmark")
    parser.add_argument("pgn", help="PGN file to sample positions from")
    parser.add_argument("--lc0", required=True, help="Path to lc0 executable")
    parser.add_argument("--large-weights", required=True, help="Path to large.pb.gz")
    parser.add_argument("--xl-weights", required=True, help="Path to extra-large.pb.gz")
    parser.add_argument("--num-positions", type=int, default=1000)
    parser.add_argument("--backend", default="cuda-fp16")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-direct-uci", action="store_true",
                        help="Use python-chess wrapper instead of direct UCI (default: direct)")
    args = parser.parse_args()

    EvaluatorClass = SyncBatchEvaluator if args.no_direct_uci else SyncDirectEvaluator
    evaluator_label = "python-chess" if args.no_direct_uci else "direct-UCI"
    print(f"[INFO] Evaluator: {evaluator_label}")

    # ── Pick positions ──────────────────────────────────────────────────
    positions = pick_middlegame_positions(args.pgn, args.num_positions, seed=args.seed)
    print(f"\n[INFO] Selected {len(positions)} positions")
    legal_counts = [len(list(chess.Board(fen).legal_moves)) for fen, _ in positions]
    print(f"  Legal moves: min={min(legal_counts)}, "
          f"mean={sum(legal_counts)/len(legal_counts):.1f}, "
          f"max={max(legal_counts)}")

    # ── Build configs ───────────────────────────────────────────────────
    # (name, weights_key, time_ms_or_None, use_exploration)
    # time_ms=None → node-based (1 node/move), time_ms=N → movetime search
    # PerPVCounters=True (exploration=True) is REQUIRED for all configs —
    # without it, MultiPV shares a single tree and movetime has no effect.
    config_defs = [
        ("large_1node",  "large", None,  True),
        ("xl_1node",     "xl",    None,  True),
        ("large_1ms",    "large", 1,     True),
        ("large_5ms",    "large", 5,     True),
        ("xl_5ms",       "xl",    5,     True),
        ("large_25ms",   "large", 25,    True),
        ("xl_25ms",      "xl",    25,    True),
        ("large_50ms",   "large", 50,    True),
        ("xl_50ms",      "xl",    50,    True),
        ("xl_1s_GT",     "xl",    1000,  True),
    ]

    weights_map = {"large": args.large_weights, "xl": args.xl_weights}

    configs = {}
    for name, wkey, time_ms, use_expl in config_defs:
        label = f"{time_ms}ms" if time_ms else "1node"
        print(f"\n[INIT] Starting engine: {name} ({label})")
        engine = EvaluatorClass(
            lc0_path=args.lc0,
            weights_path=weights_map[wkey],
            backend=args.backend,
            batch_size=args.batch_size,
            use_exploration_settings=use_expl,
            time_per_move_ms=time_ms,
        )
        engine.start()
        configs[name] = engine

    # ── Run benchmark ───────────────────────────────────────────────────
    gt_key = "xl_1s_GT"
    compare_keys = [k for k in configs if k != gt_key]

    print("\n" + "=" * 70)
    print("  RUNNING BENCHMARK")
    print("=" * 70)

    stats = compare_configs(positions, configs, ground_truth_key=gt_key)

    # ── Print results ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  RESULTS (vs {gt_key} ground truth)")
    print("  EV = W + 0.5*D from STM perspective (0.0-1.0)")
    print("=" * 70)

    print("\n  ALL LEGAL MOVES:")
    header = (f"{'Config':<15} {'Mean ΔEV':>9} {'Med ΔEV':>9} {'P95 ΔEV':>9} "
              f"{'Max ΔEV':>9} {'Rank Δ':>8} {'Top-1%':>8} {'Top-3%':>8}")
    print(header)
    print("-" * len(header))
    for name in compare_keys:
        s = stats[name]
        print(f"{name:<15} "
              f"{s['ev_mean_all']:>9.4f} "
              f"{s['ev_med_all']:>9.4f} "
              f"{s['ev_p95_all']:>9.4f} "
              f"{s['ev_max_all']:>9.4f} "
              f"{s['mean_rank_delta']:>8.2f} "
              f"{s['top1_accuracy']:>7.1f}% "
              f"{s['top3_accuracy']:>7.1f}%")

    print(f"\n  TOP-5 MOVES ONLY:")
    header2 = f"{'Config':<15} {'Mean ΔEV':>9} {'Med ΔEV':>9}"
    print(header2)
    print("-" * len(header2))
    for name in compare_keys:
        s = stats[name]
        print(f"{name:<15} "
              f"{s['ev_mean_top5']:>9.4f} "
              f"{s['ev_med_top5']:>9.4f}")

    print(f"\n  POSITION EVAL (each engine's best move):")
    header3 = f"{'Config':<15} {'Mean ΔEV':>9} {'Med ΔEV':>9} {'P95 ΔEV':>9}"
    print(header3)
    print("-" * len(header3))
    for name in compare_keys:
        s = stats[name]
        print(f"{name:<15} "
              f"{s['ev_mean_best']:>9.4f} "
              f"{s['ev_med_best']:>9.4f} "
              f"{s['ev_p95_best']:>9.4f}")

    print(f"\n  Positions: {stats[compare_keys[0]]['n_positions']} | "
          f"Moves compared: ~{stats[compare_keys[0]]['n_moves_compared']}")

    # ── Cleanup ─────────────────────────────────────────────────────────
    for engine in configs.values():
        try:
            engine.quit()
        except Exception:
            pass

    print("\n[DONE] Benchmark complete.")


if __name__ == "__main__":
    main()
