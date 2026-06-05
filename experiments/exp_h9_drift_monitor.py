"""H9: unsupervised PSI drift as an early warning before observed failures."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from secom.data import load_secom
from secom.rca import population_stability_index
from secom.results import log_result
from secom.visuals import plot_drift_curves


def make_windows(n_rows: int, window: int):
    start = 0
    while start + window <= n_rows:
        yield slice(start, start + window)
        start += window


def main() -> None:
    d = load_secom()
    X = d.X.to_numpy(dtype=float)
    y = d.y
    window = max(120, len(d.y) // 12)
    windows = list(make_windows(len(d.y), window))
    if len(windows) < 3:
        raise RuntimeError("Need at least 3 full windows for drift monitoring")

    ref = X[windows[0]]
    psi_vals = []
    fail_rates = []

    for idx, sl in enumerate(windows[1:], start=1):
        cur = X[sl]
        fail_rates.append(float(np.mean(y[sl])))
        per_feat = [
            population_stability_index(ref[:, j], cur[:, j], bins=10)
            for j in range(ref.shape[1])
            if np.isfinite(ref[:, j]).any() and np.isfinite(cur[:, j]).any()
        ]
        psi_vals.append(float(np.mean(per_feat)) if per_feat else 0.0)

    # align to same length where psi[0] is window 1 vs reference
    rho, pval = spearmanr(psi_vals, fail_rates)
    print(f"window_size={window}")
    print(f"mean PSI path: {psi_vals}")
    print(f"fail-rate path: {fail_rates}")
    print(f"spearman rho={rho:.3f}, p={pval:.4f}")

    Path("artifacts").mkdir(exist_ok=True, parents=True)
    plot_drift_curves(psi_vals, fail_rates, Path("artifacts/h9_drift_curves.png"))

    log_result(
        {
            "hypothesis": "H9",
            "name": "psi_drift",
            "mean_psi": float(np.mean(psi_vals)),
            "max_psi": float(np.max(psi_vals)),
            "spearman_rho": float(rho if np.isfinite(rho) else 0.0),
            "spearman_p": float(pval if np.isfinite(pval) else 1.0),
            "pooled_pr_auc": 0.0,
        }
    )

    print("\nVERDICT: positive monotonic drift/defect association is support for H9.")


if __name__ == "__main__":
    main()
