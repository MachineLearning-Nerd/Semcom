"""H11: test whether cycle features from deterministic timestamps add signal."""
from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from secom.data import SecomData, load_secom
from secom.experiment import run_experiment
from secom.validation import walk_forward_splits
from secom.results import log_result


def add_time_features(X, ts):
    hour = ts.dt.hour.to_numpy(dtype=float)
    dow = ts.dt.dayofweek.to_numpy(dtype=float)
    month = ts.dt.month.to_numpy(dtype=float)
    X2 = X.copy()
    X2["time_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X2["time_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    X2["time_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X2["time_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    X2["time_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    X2["time_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    return X2


def main() -> None:
    d = load_secom()
    d_time = SecomData(
        X=add_time_features(d.X, d.timestamps),
        y=d.y,
        timestamps=d.timestamps,
    )

    splits = list(walk_forward_splits(len(d.y)))
    res_base = run_experiment(d, splits, model_name="lightgbm", name="H11:base")
    res_time = run_experiment(
        d_time,
        splits,
        model_name="lightgbm",
        name="H11:time_cycles",
    )

    print(
        f"base PR-AUC={res_base.pooled_pr_auc:.3f} "
        f"Recall@25={res_base.mean_recall_at_target:.3f}"
    )
    print(
        f"time PR-AUC={res_time.pooled_pr_auc:.3f} "
        f"Recall@25={res_time.mean_recall_at_target:.3f}"
    )

    log_result(
        {
            "hypothesis": "H11",
            "name": "base",
            "pooled_pr_auc": res_base.pooled_pr_auc,
            "pooled_roc_auc": res_base.pooled_roc_auc,
            "mean_recall_at_target": res_base.mean_recall_at_target,
        }
    )
    log_result(
        {
            "hypothesis": "H11",
            "name": "time_cycles",
            "pooled_pr_auc": res_time.pooled_pr_auc,
            "pooled_roc_auc": res_time.pooled_roc_auc,
            "mean_recall_at_target": res_time.mean_recall_at_target,
        }
    )

    print("\nVERDICT: H11 holds if the cycle-features variant has higher PR-AUC or recall.")


if __name__ == "__main__":
    main()
