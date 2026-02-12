"""Shared dataset helpers for openpilot bit-flip workflows."""

from __future__ import annotations

import glob
import os
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SafeDataset(Dataset):
    """Retry nearby indices when a sample is corrupted/unreadable."""

    def __init__(self, base: Dataset, max_retry: int = 20):
        self.base = base
        self.max_retry = max_retry

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        n = len(self.base)
        for k in range(self.max_retry):
            j = (idx + k) % n
            try:
                return self.base[j]
            except Exception:
                continue
        raise RuntimeError(f"Failed to fetch a valid sample near index {idx}")


def normalize_data_root(path: str) -> str:
    return path if path.endswith("/") else path + "/"


def ensure_non_overlap_split_files(data_root: str, train_index: str, val_index: str, seed: int = 0) -> Tuple[int, int]:
    """Generate train/val segment index files if they do not exist."""
    if os.path.isfile(train_index) and os.path.isfile(val_index):
        with open(train_index, "r", encoding="utf-8") as f:
            tr_n = sum(1 for _ in f)
        with open(val_index, "r", encoding="utf-8") as f:
            va_n = sum(1 for _ in f)
        return tr_n, va_n

    pattern = os.path.join(data_root, "*", "*", "*", "video.hevc")
    sequences = glob.glob(pattern)
    if not sequences:
        raise FileNotFoundError(f"No sequences found with pattern: {pattern}")

    root = data_root.rstrip("/") + "/"
    rel = [s.replace(root, "").replace("/video.hevc", "") for s in sequences]
    route_names = sorted(set(r.split("/")[1] for r in rel))

    rng = np.random.default_rng(seed)
    rng.shuffle(route_names)
    n_train = max(1, int(0.8 * len(route_names)))
    train_routes = set(route_names[:n_train])

    train_samples = [r for r in rel if r.split("/")[1] in train_routes]
    val_samples = [r for r in rel if r.split("/")[1] not in train_routes]
    if not val_samples:
        val_samples = train_samples[-max(1, len(train_samples) // 5) :]
        train_samples = train_samples[: -len(val_samples)] or train_samples

    os.makedirs(os.path.dirname(train_index), exist_ok=True)
    with open(train_index, "w", encoding="utf-8") as f:
        f.write("\n".join(train_samples) + "\n")
    with open(val_index, "w", encoding="utf-8") as f:
        f.write("\n".join(val_samples) + "\n")

    return len(train_samples), len(val_samples)


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
