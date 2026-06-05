from __future__ import annotations

import numpy as np
import pandas as pd

from secom.rca import group_attributions, population_stability_index


def test_group_attributions_sum_by_cluster():
    shap = pd.DataFrame({"a": [1.0, -2.0], "b": [0.5, 0.5], "c": [0.0, 4.0]})
    clusters = {0: ["a", "b"], 1: ["c"]}
    g = group_attributions(shap, clusters)
    assert abs(g[0] - (abs(1) + abs(-2) + abs(0.5) + abs(0.5))) < 1e-9
    assert abs(g[1] - 4.0) < 1e-9


def test_psi_zero_for_identical_distributions():
    x = np.random.RandomState(0).normal(size=500)
    assert population_stability_index(x, x) < 1e-6
