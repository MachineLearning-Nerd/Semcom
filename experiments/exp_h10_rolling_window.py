"""H10: expanding history vs rolling window training under non-stationarity."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from secom.data import load_secom
from secom.experiment import run_experiment
from secom.results import log_result
from secom.validation import walk_forward_splits


def rolling_window_splits(
    n_samples: int,
    n_splits: int = 5,
    min_train_frac: float = 0.40,
    rolling_frac: float = 0.30,
    embargo: int = 0,
):
    start = int(n_samples * min_train_frac)
    fold_size = (n_samples - start) // n_splits
    if fold_size < 1:
        raise ValueError("too many splits")
    train_window = max(1, int(n_samples * rolling_frac))

    for k in range(n_splits):
        test_start = start + k * fold_size
        test_end = n_samples if k == n_splits - 1 else test_start + fold_size
        train_end = test_start - embargo
        if test_end <= test_start:
            continue
        train_start = max(0, train_end - train_window)
        if train_start >= train_end:
            continue
        yield np.arange(train_start, train_end, dtype=int), np.arange(test_start, test_end, dtype=int)


def main() -> None:
    d = load_secom()
    exp_splits = list(walk_forward_splits(len(d.y)))
    roll_splits = list(
        rolling_window_splits(
            len(d.y),
            n_splits=5,
            min_train_frac=0.40,
            rolling_frac=0.30,
            embargo=0,
        )
    )

    res_exp = run_experiment(d, exp_splits, model_name="lightgbm", name="H10:expanding")
    res_roll = run_experiment(
        d, roll_splits, model_name="lightgbm", name="H10:rolling30pct"
    )

    print(f"expanding PR-AUC={res_exp.pooled_pr_auc:.3f} ROC-AUC={res_exp.pooled_roc_auc:.3f}")
    print(f"rolling   PR-AUC={res_roll.pooled_pr_auc:.3f} ROC-AUC={res_roll.pooled_roc_auc:.3f}")

    log_result(
        {
            "hypothesis": "H10",
            "name": "expanding",
            "pooled_pr_auc": res_exp.pooled_pr_auc,
            "pooled_roc_auc": res_exp.pooled_roc_auc,
            "mean_recall_at_target": res_exp.mean_recall_at_target,
            "n_folds": res_exp.meta.get("n_folds", 0),
        }
    )
    log_result(
        {
            "hypothesis": "H10",
            "name": "rolling30",
            "pooled_pr_auc": res_roll.pooled_pr_auc,
            "pooled_roc_auc": res_roll.pooled_roc_auc,
            "mean_recall_at_target": res_roll.mean_recall_at_target,
            "n_folds": res_roll.meta.get("n_folds", 0),
        }
    )

    winner = "expanding"
    if res_roll.pooled_pr_auc > res_exp.pooled_pr_auc:
        winner = "rolling30"
    print(f"\nVERDICT: H10 holds if rolling10 wins; observed winner={winner}.")


if __name__ == "__main__":
    main()
