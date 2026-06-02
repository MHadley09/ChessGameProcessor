import numpy as np, random
d = np.load(r"C:\Users\micha\Personal\School\DEng\dissertation\mutation\ChessGameProcessor\dataset\v5\train\shard_w02_0000.npz", allow_pickle=True)
i = random.randint(0, len(d["fen_before"])-1)
m = int(d["possible_mask"][i].sum())
print(f"idx={i} fen={d['fen_before'][i]}")
print(f"played={d['possible_uci'][i][d['actual_idx'][i]]} actual_idx={d['actual_idx'][i]} mistake={d['is_mistake'][i]:.0f} phase={d['game_phase'][i] if 'game_phase' in d else 'N/A'}")
print(f"tabular={d['tabular'][i]}")
print(f"wdl_before={d['win_prob_before'][i]} time_log={d['time_spent_log'][i]:.3f}")
print(f"--- {m} legal moves ---")
for j in range(m):
    print(f"  {j}: {d['possible_uci'][i][j]:7s} eval={d['possible_scalars'][i][j][0]:+8.1f} wdl=[{d['possible_scalars'][i][j][1]:.1f},{d['possible_scalars'][i][j][2]:.1f},{d['possible_scalars'][i][j][3]:.1f}]")
