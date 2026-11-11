"""Training windows sliced lazily from the memory-mapped vector cache.

The memmaps are (re)opened inside each DataLoader worker, so datasets pickle
cheaply under the ``spawn`` start method (Windows / CUDA) instead of copying
the arrays into every worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.device import DeviceContext


@dataclass
class WindowSpec:
    context: int
    horizon: int  # supervised days (e.g. 90)
    horizon_pad: int  # model output length (multiple of patch length)


class CacheWindowDataset(Dataset):
    def __init__(self, cache_dir: Path, index: np.ndarray, spec: WindowSpec):
        """``index`` is ``[n, 2]`` of (account row, cutoff day) where the cutoff
        is the first forecast day; context = ``[cutoff - L, cutoff)``."""
        self.cache_dir = Path(cache_dir)
        self.index = index.astype(np.int64)
        self.spec = spec
        self._past = None
        self._known = None

    def _open(self) -> None:
        if self._past is None:
            self._past = np.load(self.cache_dir / "past.npy", mmap_mode="r")
            self._known = np.load(self.cache_dir / "known.npy", mmap_mode="r")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_past"] = state["_known"] = None
        return state

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        self._open()
        row, cut = self.index[i]
        L, H, Hp = self.spec.context, self.spec.horizon, self.spec.horizon_pad
        lo = cut - L
        past = np.full((L, self._past.shape[2]), np.nan, dtype=np.float32)
        known = np.zeros((L + Hp, self._known.shape[2]), dtype=np.float32)
        s = max(lo, 0)
        past[s - lo :] = self._past[row, s:cut]
        known[s - lo :] = self._known[row, s : cut + Hp]
        future = np.full((Hp, 2), np.nan, dtype=np.float32)
        end = min(cut + H, self._past.shape[1])
        future[: end - cut] = self._past[row, cut:end, :2]
        return torch.from_numpy(past), torch.from_numpy(known), torch.from_numpy(future)


def holdout_split(n_accounts: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(validation rows, training rows) of the account-level holdout."""
    perm = np.random.default_rng(seed).permutation(n_accounts)
    n_val = int(round(n_accounts * val_fraction)) if n_accounts > 1 else 0
    return perm[:n_val], perm[n_val:]


def split_rows_by_group(
    groups: np.ndarray, val_fraction: float, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train, validation, test) rows where every group (e.g. a customer owning several cards) lands
    wholly in one of the three sets."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    n_test, n_val = int(round(len(uniq) * test_fraction)), int(round(len(uniq) * val_fraction))
    is_test = np.isin(groups, uniq[perm[:n_test]])
    is_val = np.isin(groups, uniq[perm[n_test : n_test + n_val]])
    rows = np.arange(len(groups))
    return rows[~is_test & ~is_val], rows[is_val], rows[is_test]


def build_window_index(
    n_accounts: int,
    n_days: int,
    spec: WindowSpec,
    stride: int,
    val_fraction: float,
    seed: int,
    min_context: int | None = None,
    max_target_day: int | None = None,
    row_split: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Account-level holdout: validation accounts are never seen in training.

    ``row_split`` = (train rows, validation rows) replaces the random account split, e.g. to split by
    customer so one customer's cards never sit on both sides.

    ``max_target_day`` (inclusive day index) drops every window whose
    supervised horizon would reach past it, for strictly out-of-time
    evaluation after that day."""
    min_ctx = min_context or spec.context // 2
    last_cut = n_days - spec.horizon
    if max_target_day is not None:
        last_cut = min(last_cut, max_target_day - spec.horizon + 1)
    cuts = np.arange(min_ctx, last_cut + 1, stride)
    if cuts.size == 0:
        raise ValueError(f"need at least {min_ctx + spec.horizon} days of history to train, have {n_days}")
    if row_split is not None:
        train_rows, val_rows = (np.asarray(a, dtype=np.int64) for a in row_split)
        if np.intersect1d(train_rows, val_rows).size:
            raise ValueError("row_split: a row is in both train and validation")
    else:
        val_rows, train_rows = holdout_split(n_accounts, val_fraction, seed)

    def grid(rows: np.ndarray, cut_set: np.ndarray) -> np.ndarray:
        if rows.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        r, c = np.meshgrid(rows, cut_set, indexing="ij")
        return np.stack([r.ravel(), c.ravel()], axis=1)

    # validate on the most recent cutoffs (the regime the EWS is scored on)
    val_cuts = cuts[-max(1, len(cuts) // 4) :]
    return grid(train_rows, cuts), grid(val_rows, val_cuts)


def make_loader(
    ds: CacheWindowDataset, batch_size: int, shuffle: bool, ctx: DeviceContext, num_workers: int
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle and len(ds) > batch_size,
        num_workers=num_workers,
        pin_memory=ctx.pin_memory,  # page-locked host buffers -> async H2D copies on CUDA
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
