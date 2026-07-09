"""
vet_dataset.py — Spot-check a handful of random positions from produced NPZ shards.
Prints game metadata, tabular features, possible moves with scalars, and flags.

Usage:
    python vet_dataset.py <shard_dir_or_file> [--n 5] [--seed 42]
"""

import numpy as np
import chess
import os, sys, glob, argparse, random

TABULAR_LABELS = [
    "time_remaining/3600",    # 0
    "w_elo/3000",             # 1
    "b_elo/3000",             # 2
    "elo_diff/1000",          # 3
    "move_no/200",            # 4
    "is_white",               # 5
    "eval_stm/1000",          # 6
    "wdl_w_before",           # 7
    "wdl_d_before",           # 8
    "wdl_l_before",           # 9
    "init_time/3600",         # 10
    "increment/60",           # 11
    "prev_capture",           # 12
    "in_check",               # 13
    "eval_std/1000",          # 14
    "frac_captures",          # 15
    "frac_checks",            # 16
    "frac_candidates",        # 17
]

POSS_SCALAR_LABELS = [
    "eval_stm/1000",    # 0
    "wdl_w",            # 1
    "wdl_d",            # 2
    "wdl_l",            # 3
    "log1p_nodes/20",   # 4
    "depth/40",         # 5
    "move_quality",     # 6
    "piece_val",        # 7
    "is_capture",       # 8
    "is_check",         # 9
    "is_checkmate",     # 10
    "policy(unused)",   # 11
]

PIECE_VAL_MAP = {1/6: 'P', 2/6: 'N', 3/6: 'B', 4/6: 'R', 5/6: 'Q', 1.0: 'K'}

def closest_piece(val):
    best = min(PIECE_VAL_MAP.keys(), key=lambda k: abs(k - val))
    return PIECE_VAL_MAP[best] if abs(best - val) < 0.01 else f"?({val:.3f})"

def phase_from_fen(fen, move_no):
    """Match dataset_writer._compute_game_phase logic using FEN + move number."""
    try:
        board = chess.Board(fen)
    except Exception:
        # Fallback to ply-only
        if move_no <= 14: return "opening"
        return "middlegame"

    queens_w = len(board.pieces(chess.QUEEN, chess.WHITE))
    queens_b = len(board.pieces(chess.QUEEN, chess.BLACK))
    mm_w = (len(board.pieces(chess.KNIGHT, chess.WHITE)) +
            len(board.pieces(chess.BISHOP, chess.WHITE)) +
            len(board.pieces(chess.ROOK, chess.WHITE)))
    mm_b = (len(board.pieces(chess.KNIGHT, chess.BLACK)) +
            len(board.pieces(chess.BISHOP, chess.BLACK)) +
            len(board.pieces(chess.ROOK, chess.BLACK)))

    total_q = queens_w + queens_b
    total_mm = mm_w + mm_b

    is_endgame = False
    if total_q == 0:
        is_endgame = True
    elif total_q == 1:
        if queens_w == 1 and mm_w == 0:
            is_endgame = True
        elif queens_b == 1 and mm_b == 0:
            is_endgame = True
    if total_q == 0 and total_mm <= 3:
        is_endgame = True

    if is_endgame: return "endgame"
    if move_no <= 14: return "opening"
    return "middlegame"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="NPZ shard file or directory of shards")
    parser.add_argument("--n", type=int, default=5, help="Number of positions to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Gather shard files
    if os.path.isdir(args.path):
        shards = sorted(glob.glob(os.path.join(args.path, "**/*.npz"), recursive=True))
    else:
        shards = [args.path]

    if not shards:
        print("No .npz files found"); return

    random.seed(args.seed)

    # Collect all examples across shards with their shard path
    all_examples = []
    for shard_path in shards:
        try:
            data = np.load(shard_path, allow_pickle=True)
            n_examples = len(data['tabular'])
            for i in range(n_examples):
                all_examples.append((shard_path, data, i))
        except Exception as e:
            print(f"[WARN] Couldn't load {shard_path}: {e}")

    if not all_examples:
        print("No examples found"); return

    print(f"Loaded {len(all_examples)} examples from {len(shards)} shard(s)\n")

    picks = random.sample(all_examples, min(args.n, len(all_examples)))

    for pick_num, (shard_path, data, idx) in enumerate(picks, 1):
        tab = data['tabular'][idx]
        poss_scalars = data['possible_scalars'][idx]
        poss_mask = data['possible_mask'][idx]
        actual_idx = int(data['actual_idx'][idx])
        is_mistake = float(data['is_mistake'][idx])
        wdl_before = data['win_prob_before'][idx]
        time_spent = float(data['time_spent_log'][idx])
        fen = str(data['fen_before'][idx]) if 'fen_before' in data else "N/A"
        poss_uci = data['possible_uci'][idx] if 'possible_uci' in data else None

        n_legal = int(poss_mask.sum())

        # Decode tabular
        w_elo = tab[1] * 3000
        b_elo = tab[2] * 3000
        move_no = tab[4] * 200
        is_white = tab[5] > 0.5
        eval_stm = tab[6] * 1000

        phase = phase_from_fen(fen, int(move_no))

        # Classify played move
        if actual_idx == 0:
            move_class = "★ EXCELLENT"
        elif actual_idx <= 2:
            move_class = "✓ GOOD (top-3)"
        elif actual_idx <= 4:
            move_class = "~ OK (top-5)"
        elif is_mistake > 0.5:
            move_class = "✗ MISTAKE"
        else:
            move_class = f"  rank #{actual_idx+1}"

        print("=" * 80)
        print(f"  EXAMPLE {pick_num} — Shard: {os.path.basename(shard_path)}, Index: {idx}")
        print("=" * 80)
        print(f"  FEN: {fen}")
        print(f"  Side: {'White' if is_white else 'Black'} | Move#: {int(move_no)} | Phase: {phase}")
        print(f"  Elo: W={w_elo:.0f} B={b_elo:.0f} (diff={tab[3]*1000:+.0f})")
        print(f"  Eval (STM): {eval_stm:.1f} cp")
        print(f"  WDL before: W={wdl_before[0]:.3f} D={wdl_before[1]:.3f} L={wdl_before[2]:.3f}")
        print(f"  In check: {bool(tab[13])} | Prev capture: {bool(tab[12])}")
        print(f"  Time remaining: {tab[0]*3600:.0f}s | Time spent: e^{time_spent:.2f}-1 = {np.expm1(time_spent):.1f}s")
        print(f"  Time control: {tab[10]*3600:.0f}s + {tab[11]*60:.0f}s")
        print(f"  Legal moves: {n_legal} | Eval std: {tab[14]*1000:.1f} cp")
        print(f"  Frac captures: {tab[15]:.2f} | Frac checks: {tab[16]:.2f}")
        frac_mist = tab[18] if len(tab) > 18 else float('nan')
        frac_exc = tab[19] if len(tab) > 19 else float('nan')
        print(f"  Frac mistakes: {frac_mist:.2f} | Frac excellent: {frac_exc:.2f}")
        print(f"  Actual move idx: {actual_idx} | Is mistake: {bool(is_mistake)} | {move_class}")

        print(f"\n  {'#':>3} {'UCI':>7} {'Piece':>5} {'Eval':>8} {'W':>6} {'D':>6} {'L':>6} {'Nodes':>6} {'Dep':>4} {'Qual':>5} {'Cap':>3} {'Chk':>3} {'Mt':>3} {'Played':>6}")
        print("  " + "-" * 76)

        for i in range(n_legal):
            s = poss_scalars[i]
            uci = poss_uci[i] if poss_uci is not None else "?"
            ev = s[0] * 1000
            ww, wd, wl = s[1], s[2], s[3]
            nodes = np.expm1(s[4] * 20)
            depth = s[5] * 40
            qual = s[6]
            piece = closest_piece(s[7])
            cap = "Y" if s[8] > 0.5 else ""
            chk = "Y" if s[9] > 0.5 else ""
            mate = "Y" if s[10] > 0.5 else ""
            played = " <<<" if i == actual_idx else ""

            print(f"  {i+1:3} {uci:>7} {piece:>5} {ev:>8.1f} {ww:>6.3f} {wd:>6.3f} {wl:>6.3f} {nodes:>6.0f} {depth:>4.0f} {qual:>5.2f} {cap:>3} {chk:>3} {mate:>3} {played}")

        # Sanity checks
        issues = []
        unique_wdls = len(set((poss_scalars[i,1], poss_scalars[i,2], poss_scalars[i,3]) for i in range(n_legal)))
        if unique_wdls < n_legal * 0.5 and n_legal > 5:
            issues.append(f"Only {unique_wdls}/{n_legal} unique WDLs — possible eval collapse")
        if actual_idx < 0:
            issues.append("Played move not found in possibles!")
        if wdl_before.sum() < 0.9 or wdl_before.sum() > 1.1:
            issues.append(f"WDL before doesn't sum to ~1.0: {wdl_before.sum():.3f}")
        if abs(tab[7] + tab[8] + tab[9] - 1.0) > 0.1:
            issues.append(f"Tabular WDL doesn't sum to ~1.0: {tab[7]+tab[8]+tab[9]:.3f}")

        if issues:
            print(f"\n  ⚠️  ISSUES:")
            for iss in issues:
                print(f"     - {iss}")

        print()

    # ── Aggregate summary ──────────────────────────────────────────────
    print("=" * 80)
    print("  AGGREGATE SUMMARY")
    print("=" * 80)

    n_total = len(picks)
    n_mistakes = sum(1 for _, d, i in picks if float(d['is_mistake'][i]) > 0.5)
    n_excellent = sum(1 for _, d, i in picks if int(d['actual_idx'][i]) == 0)
    n_top3 = sum(1 for _, d, i in picks if 0 <= int(d['actual_idx'][i]) <= 2)
    n_top5 = sum(1 for _, d, i in picks if 0 <= int(d['actual_idx'][i]) <= 4)
    n_top10 = sum(1 for _, d, i in picks if 0 <= int(d['actual_idx'][i]) <= 9)
    n_missing = sum(1 for _, d, i in picks if int(d['actual_idx'][i]) < 0)

    avg_idx = np.mean([int(d['actual_idx'][i]) for _, d, i in picks if int(d['actual_idx'][i]) >= 0])
    avg_legal = np.mean([float(d['possible_mask'][i].sum()) for _, d, i in picks])

    # WDL collapse rate
    collapse_positions = 0
    for _, d, i in picks:
        mask = d['possible_mask'][i]
        ps = d['possible_scalars'][i]
        nl = int(mask.sum())
        if nl > 5:
            uwdl = len(set((ps[j,1], ps[j,2], ps[j,3]) for j in range(nl)))
            if uwdl < nl * 0.5:
                collapse_positions += 1

    # Phase distribution
    phase_counts = {'opening': 0, 'middlegame': 0, 'endgame': 0}
    for _, d, i in picks:
        fen_p = str(d['fen_before'][i]) if 'fen_before' in d else ""
        mn = d['tabular'][i][4] * 200
        ph = phase_from_fen(fen_p, int(mn))
        phase_counts[ph] += 1

    # Mistake by phase
    mistake_by_phase = {'opening': [0, 0], 'middlegame': [0, 0], 'endgame': [0, 0]}
    for _, d, i in picks:
        fen_p = str(d['fen_before'][i]) if 'fen_before' in d else ""
        mn = d['tabular'][i][4] * 200
        ph = phase_from_fen(fen_p, int(mn))
        mistake_by_phase[ph][1] += 1
        if float(d['is_mistake'][i]) > 0.5:
            mistake_by_phase[ph][0] += 1

    print(f"  Positions sampled:    {n_total}")
    print(f"  Avg legal moves:      {avg_legal:.1f}")
    print(f"  Avg played move rank: {avg_idx:.1f} (0-indexed)")
    print()
    print(f"  Excellent (rank #1):  {n_excellent:>4} / {n_total}  ({n_excellent/n_total*100:.1f}%)")
    print(f"  Top-3:                {n_top3:>4} / {n_total}  ({n_top3/n_total*100:.1f}%)")
    print(f"  Top-5:                {n_top5:>4} / {n_total}  ({n_top5/n_total*100:.1f}%)")
    print(f"  Top-10:               {n_top10:>4} / {n_total}  ({n_top10/n_total*100:.1f}%)")
    print(f"  Mistakes:             {n_mistakes:>4} / {n_total}  ({n_mistakes/n_total*100:.1f}%)")
    print(f"  Move not found:       {n_missing:>4} / {n_total}")
    print(f"  WDL collapse (>50%):  {collapse_positions:>4} / {n_total}  ({collapse_positions/n_total*100:.1f}%)")
    print()
    print(f"  Phase distribution:")
    for ph in ['opening', 'middlegame', 'endgame']:
        cnt = phase_counts[ph]
        m, t = mistake_by_phase[ph]
        mrate = f"{m/t*100:.1f}%" if t > 0 else "N/A"
        print(f"    {ph:>12}: {cnt:>4} positions, {mrate} mistake rate")
    print()

if __name__ == "__main__":
    main()
