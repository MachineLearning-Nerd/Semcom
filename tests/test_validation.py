"""Executable spec for walk_forward_splits (TDD).

These tests encode the no-future-leakage contract. They will FAIL until
`walk_forward_splits` is implemented -- that's the point: make them green.
"""
from __future__ import annotations

import numpy as np
import pytest

from secom.validation import walk_forward_splits

N = 1000
N_SPLITS = 5


def folds():
    return list(walk_forward_splits(N, n_splits=N_SPLITS, min_train_frac=0.4))


def test_produces_requested_number_of_folds():
    assert len(folds()) == N_SPLITS


def test_no_future_leakage():
    """Every training index must precede every test index in the same fold."""
    for train_idx, test_idx in folds():
        assert train_idx.max() < test_idx.min()


def test_embargo_is_respected():
    embargo = 10
    for train_idx, test_idx in walk_forward_splits(
        N, n_splits=N_SPLITS, min_train_frac=0.4, embargo=embargo
    ):
        assert test_idx.min() - train_idx.max() > embargo


def test_training_window_expands():
    """Each fold's training set strictly contains the previous fold's."""
    prev_max = -1
    for train_idx, _ in folds():
        assert train_idx.min() == 0          # expanding window starts at 0
        assert train_idx.max() > prev_max    # grows every fold
        prev_max = train_idx.max()


def test_test_folds_are_disjoint_and_cover_tail():
    test_blocks = [test_idx for _, test_idx in folds()]
    all_test = np.concatenate(test_blocks)
    assert len(all_test) == len(np.unique(all_test))           # disjoint
    assert all_test.max() == N - 1                             # reaches the end
    assert all_test.min() >= int(N * 0.4)                      # respects warm-up


def test_indices_are_valid_and_train_test_disjoint():
    for train_idx, test_idx in folds():
        assert train_idx.min() >= 0 and test_idx.max() < N
        assert len(np.intersect1d(train_idx, test_idx)) == 0


@pytest.mark.parametrize("n_splits", [3, 4, 6, 8])
def test_works_for_various_split_counts(n_splits):
    result = list(walk_forward_splits(N, n_splits=n_splits, min_train_frac=0.4))
    assert len(result) == n_splits
