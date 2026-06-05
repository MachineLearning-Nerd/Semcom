"""Walk-forward (time-respecting) cross-validation splits.

This is the methodological heart of the PoC. Because the SECOM fail rate
drifts (~14% early -> ~3.5% late), a random k-fold trains on the future
and tests on the past, inflating and destabilizing every metric. Walk-
forward validation forbids that: each fold trains only on rows that came
*before* the rows it is tested on.

>>> The implementation of `walk_forward_splits` is intentionally left to
>>> you -- see the TODO. `tests/test_validation.py` is its executable
>>> specification; make those tests pass.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import List, Tuple

import numpy as np

from . import config


def walk_forward_splits(
    n_samples: int,
    n_splits: int = config.N_SPLITS,
    min_train_frac: float = config.MIN_TRAIN_FRAC,
    embargo: int = config.EMBARGO,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) pairs for expanding-window validation.

    Assumes rows 0..n_samples-1 are already in chronological order.

    Contract (verified by tests/test_validation.py):
      * Chronology / no leakage: every train index < every test index in
        the same fold (with at least ``embargo`` rows purged between them).
      * Expanding window: the training set grows monotonically across folds
        (fold k's train is a prefix that contains fold k-1's train).
      * Coverage: the union of all test folds covers the tail of the data
        after the initial ``min_train_frac`` warm-up, with disjoint folds.
      * Validity: indices are within [0, n_samples) and train/test are
        disjoint within each fold.

    Args:
        n_samples: total number of (time-ordered) rows.
        n_splits: number of train/test folds to produce.
        min_train_frac: fraction of rows reserved as the first training
            window before any testing begins.
        embargo: number of rows to skip between the end of train and the
            start of test, to purge autocorrelation leakage.

    Yields:
        (train_idx, test_idx) as integer numpy arrays.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if not 0 < min_train_frac < 1:
        raise ValueError("min_train_frac must be in (0, 1)")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if int(n_samples * min_train_frac) < 1:
        raise ValueError("min_train_frac leaves no training rows")

    start = int(n_samples * min_train_frac)
    if start < 1 or start >= n_samples:
        raise ValueError("min_train_frac/n_splits leave no room for folds")

    fold_size = (n_samples - start) // n_splits
    if fold_size < 1:
        raise ValueError("too many splits for available tail")

    for k in range(n_splits):
        test_start = start + k * fold_size
        test_end = n_samples if k == n_splits - 1 else test_start + fold_size
        train_end = test_start - embargo
        if train_end < 1:
            raise ValueError("embargo too large for first fold")
        if test_end <= test_start:
            continue

        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)
        if len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def random_stratified_splits(
    y: np.ndarray,
    n_splits: int = config.N_SPLITS,
) -> List[tuple[np.ndarray, np.ndarray]]:
    """LEAKY-by-design random stratified baseline (used only for H1)."""
    from sklearn.model_selection import StratifiedKFold

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if len(np.asarray(y)) < n_splits:
        raise ValueError("y is too short for requested split count")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    dummy = np.zeros((len(y), 1))
    return [(tr, te) for tr, te in skf.split(dummy, y)]
