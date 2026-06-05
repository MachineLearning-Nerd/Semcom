from __future__ import annotations

import numpy as np
import pytest

from secom.models import make_model, fail_scores


def _can_use_lgbm():
    try:
        make_model("lightgbm")
    except Exception:
        return False
    return True


def test_every_model_emits_calibrated_fail_scores():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(0, 1, (80, 5)), rng.normal(3, 1, (20, 5))])
    y = np.r_[np.zeros(80, int), np.ones(20, int)]
    for name in ["logreg", "rf", "iforest"]:
        m = make_model(name)
        if name == "iforest":
            m.fit(X)
        else:
            m.fit(X, y)
        s = fail_scores(m, X)
        assert s.shape == (100,)
        assert s[80:].mean() > s[:80].mean()

    if _can_use_lgbm():
        m = make_model("lightgbm")
        m.fit(X, y)
        s = fail_scores(m, X)
        assert s.shape == (100,)
        assert s[80:].mean() > s[:80].mean()
    else:
        pytest.skip("lightgbm unavailable")
