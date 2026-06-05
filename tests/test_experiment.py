from __future__ import annotations

import numpy as np
import pandas as pd

from secom.data import SecomData
from secom.experiment import run_experiment
from secom.validation import walk_forward_splits
from secom.preprocess import Preprocessor


def _toy(n=600, p=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + rng.normal(0, 0.5, n) > 1.0).astype(int)
    ts = pd.Series(pd.date_range("2008-07-19", periods=n, freq="h"))
    return SecomData(X=pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y=y, timestamps=ts)


def test_runner_is_leakfree_and_aggregates():
    d = _toy()
    splits = list(walk_forward_splits(len(d.y), n_splits=4))
    res = run_experiment(
        d,
        splits,
        model_name="logreg",
        make_preprocessor=lambda: Preprocessor(missing_indicator_min_gap=2.0),
    )
    assert len(res.fold_scores) == 4
    assert res.oof_scores.shape == res.oof_y.shape
    assert 0.5 < res.pooled_pr_auc <= 1.0
    assert res.mean_roc_auc > 0.7
