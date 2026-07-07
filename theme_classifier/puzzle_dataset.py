#!/usr/bin/env python3
"""
Puzzle PyTorch Dataset
======================
Loads NPZ shards produced by puzzle_dataset_processor.py for training
the theme classifier and opening classifier.

Supports:
- Multi-label theme targets (BCEWithLogitsLoss)
- Multi-class opening targets (CrossEntropyLoss)
- Shard-aware sampling (ShardGroupSampler compatible)
- Train/val/test splits by shard
- Optional filtering (themes only, openings only)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PuzzleDataset(Dataset):
    """PyTorch dataset for puzzle theme/opening classification.
    
    Loads NPZ shards lazily (one shard at a time in memory).
    Compatible with V5's ShardGroupSampler for efficient I/O.
    
    Args:
        data_dir: Path to directory containing NPZ shards
        split: "train", "val", or "test" (uses split_config.json)
        mode: "theme" (multi-label), "opening" (multi-class), or "both"
        opening_labels_path: Path to opening_labels.json
        transform: Optional transform for board planes
    """

    def __init__(
        self,
        data_dir: str,
        split: Optional[str] = None,
        mode: str = "both",
        opening_labels_path: Optional[str] = None,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.shards_dir = self.data_dir / "shards"
        self.mode = mode
        self.transform = transform

        # Find all shard files
        self.shard_paths = sorted(self.shards_dir.glob("puzzle_*.npz"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard files found in {self.shards_dir}")

        # Load split config if available
        split_config_path = self.data_dir / "split_config.json"
        if split and split_config_path.exists():
            with open(split_config_path) as f:
                split_config = json.load(f)
            split_shard_names = set(split_config.get(split, []))
            self.shard_paths = [
                p for p in self.shard_paths if p.name in split_shard_names
            ]

        # Build index: (shard_idx, example_idx_within_shard)
        self._index: List[Tuple[int, int]] = []
        self._shard_sizes: List[int] = []
        self._shard_groups: List[List[int]] = []  # for ShardGroupSampler

        for shard_idx, shard_path in enumerate(self.shard_paths):
            # Quick peek at shard size without loading data
            with np.load(str(shard_path), allow_pickle=True) as npz:
                n = len(npz["current_planes"])
            self._shard_sizes.append(n)
            group = []
            for i in range(n):
                global_idx = len(self._index)
                self._index.append((shard_idx, i))
                group.append(global_idx)
            self._shard_groups.append(group)

        self._total = len(self._index)

        # Cache: loaded shard data
        self._cached_shard_idx = -1
        self._cached_data: Optional[Dict[str, np.ndarray]] = None

        # Opening label mapping
        self.opening_names = {}
        if opening_labels_path and os.path.exists(opening_labels_path):
            with open(opening_labels_path) as f:
                olabels = json.load(f)
            self.opening_names = {
                int(k): v for k, v in olabels.get("idx_to_tag", {}).items()
            }
            self.n_openings = len(self.opening_names)
        else:
            self.n_openings = 0

        # Theme info
        theme_labels_path = self.data_dir / "theme_labels.json"
        if theme_labels_path.exists():
            with open(theme_labels_path) as f:
                tinfo = json.load(f)
            self.theme_names = tinfo.get("themes", [])
            self.n_themes = len(self.theme_names)
        else:
            self.theme_names = []
            self.n_themes = 62  # default

    def __len__(self) -> int:
        return self._total

    def _load_shard(self, shard_idx: int):
        """Load a shard into the cache."""
        if shard_idx == self._cached_shard_idx:
            return
        shard_path = self.shard_paths[shard_idx]
        self._cached_data = dict(np.load(str(shard_path), allow_pickle=True))
        self._cached_shard_idx = shard_idx

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, local_idx = self._index[idx]
        self._load_shard(shard_idx)
        d = self._cached_data

        # Board planes
        planes = torch.from_numpy(d["current_planes"][local_idx].copy())
        if self.transform:
            planes = self.transform(planes)

        result = {"board_planes": planes}

        # Theme targets (multi-label)
        if self.mode in ("theme", "both"):
            themes = torch.from_numpy(d["themes"][local_idx].copy())
            result["theme_targets"] = themes

        # Opening targets (multi-class, single int label)
        if self.mode in ("opening", "both"):
            opening_idx = int(d["opening_idx"][local_idx])
            result["opening_target"] = torch.tensor(opening_idx, dtype=torch.long)
            result["has_opening"] = torch.tensor(
                1.0 if opening_idx >= 0 else 0.0, dtype=torch.float32
            )

        # V5-compatible fields (for joint training or fine-tuning)
        if "possible_scalars" in d:
            result["possible_scalars"] = torch.from_numpy(
                d["possible_scalars"][local_idx].copy()
            )
        if "possible_mask" in d:
            result["possible_mask"] = torch.from_numpy(
                d["possible_mask"][local_idx].copy()
            )
        if "tabular" in d:
            result["tabular"] = torch.from_numpy(d["tabular"][local_idx].copy())
        if "actual_idx" in d:
            result["actual_idx"] = torch.tensor(
                int(d["actual_idx"][local_idx]), dtype=torch.long
            )
        if "is_mistake" in d:
            result["is_mistake"] = torch.tensor(
                float(d["is_mistake"][local_idx]), dtype=torch.float32
            )
        if "win_prob_before" in d:
            result["win_prob_before"] = torch.from_numpy(
                d["win_prob_before"][local_idx].copy()
            )

        # Puzzle metadata
        if "puzzle_rating" in d:
            result["puzzle_rating"] = torch.tensor(
                int(d["puzzle_rating"][local_idx]), dtype=torch.int32
            )

        return result

    @property
    def shard_groups(self) -> List[List[int]]:
        """For ShardGroupSampler compatibility."""
        return self._shard_groups


# ---------------------------------------------------------------------------
# Shard Group Sampler (same pattern as V5)
# ---------------------------------------------------------------------------
class ShardGroupSampler(Sampler):
    """Samples indices grouped by shard for sequential I/O.
    
    Within each shard, indices are shuffled. Shard order is also shuffled.
    This matches the V5 training DataLoader pattern.
    """

    def __init__(self, dataset: PuzzleDataset, seed: int = 42):
        self.groups = dataset.shard_groups
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._epoch)

        # Shuffle shard order
        group_order = list(range(len(self.groups)))
        rng.shuffle(group_order)

        for g in group_order:
            indices = list(self.groups[g])
            rng.shuffle(indices)
            yield from indices

    def __len__(self):
        return sum(len(g) for g in self.groups)


# ---------------------------------------------------------------------------
# Split Generator
# ---------------------------------------------------------------------------
def create_train_val_test_split(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """Create a split_config.json assigning shards to train/val/test.
    
    Split is by shard (not by example) for clean separation.
    """
    shards_dir = Path(data_dir) / "shards"
    shard_names = sorted([p.name for p in shards_dir.glob("puzzle_*.npz")])
    n = len(shard_names)
    if n == 0:
        raise FileNotFoundError(f"No shards in {shards_dir}")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # n_test = n - n_train - n_val

    split = {
        "train": [shard_names[i] for i in perm[:n_train]],
        "val": [shard_names[i] for i in perm[n_train:n_train + n_val]],
        "test": [shard_names[i] for i in perm[n_train + n_val:]],
    }

    config_path = Path(data_dir) / "split_config.json"
    with open(config_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Split created: train={len(split['train'])}, "
          f"val={len(split['val'])}, test={len(split['test'])} shards")
    return split


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------
def puzzle_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate for variable-presence keys."""
    result = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[k] = torch.stack(vals)
        else:
            result[k] = vals
    return result
