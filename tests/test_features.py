from __future__ import annotations

import numpy as np
import pandas as pd

from secom.features import UnivariateScreen, ClusterReducer


def test_univariate_screen_fits_on_train_only():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"f{i}" for i in range(10)])
    y = (X["f0"] > 0).astype(int).to_numpy()
    scr = UnivariateScreen(k=3).fit(X, y)
    assert "f0" in scr.selected_ and len(scr.selected_) == 3
    assert list(scr.transform(X).columns) == scr.selected_


def test_cluster_reducer_drops_duplicate_columns():
    rng = np.random.RandomState(1)
    base = rng.normal(size=(100, 1))
    X = pd.DataFrame(np.hstack([base, base, rng.normal(size=(100, 1))]), columns=["a", "a_dup", "b"])
    red = ClusterReducer(corr_threshold=0.99).fit(X)
    out = red.transform(X)
    assert out.shape[1] == 2
