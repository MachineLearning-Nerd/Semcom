"""H6: compare model families and take screened+clustered LightGBM as candidate.

This task establishes the headline detector family by testing a broad set of
algorithms under the same walk-forward split regime.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from secom.data import load_secom
from secom.experiment import run_experiment
from secom.features import ClusterReducer, UnivariateScreen
from secom.results import log_result
from secom.validation import walk_forward_splits


def screened_cluster_fit_transform(pp, Xtr, ytr):
    """Fit univariate screening + cluster reduction on train only."""
    screen = UnivariateScreen(k=120)
    Xtr_screen = screen.fit_transform(Xtr, ytr)
    reducer = ClusterReducer(corr_threshold=0.98, method="mean").fit(Xtr_screen)
    Xr = reducer.transform(Xtr_screen)

    def transform(Xte):
        return reducer.transform(screen.transform(Xte))

    return Xr, transform


def main() -> None:
    d = load_secom()
    splits = list(walk_forward_splits(len(d.y)))

    results = {}
    results["logreg_all"] = run_experiment(
        d, splits, model_name="logreg", name="H6:logreg_all"
    )
    results["rf_all"] = run_experiment(
        d, splits, model_name="rf", name="H6:rf_all"
    )
    results["lgbm_all"] = run_experiment(
        d, splits, model_name="lightgbm", name="H6:lgbm_all"
    )
    results["lgbm_screened_clustered"] = run_experiment(
        d,
        splits,
        model_name="lightgbm",
        fit_transform=screened_cluster_fit_transform,
        name="H6:lgbm_screened_clustered",
    )

    for name, res in results.items():
        print(
            f"{name:28} PR-AUC={res.pooled_pr_auc:.3f} "
            f"ROC-AUC={res.pooled_roc_auc:.3f} "
            f"Recall@25={res.mean_recall_at_target:.3f}"
        )
        log_result(
            {
                "hypothesis": "H6",
                "name": name,
                "pooled_pr_auc": res.pooled_pr_auc,
                "pooled_roc_auc": res.pooled_roc_auc,
                "mean_recall_at_target": res.mean_recall_at_target,
            }
        )

    winner = max(results.items(), key=lambda kv: kv[1].pooled_pr_auc)
    print(
        "\nVERDICT: Headline detector = "
        f"{winner[0]} with PR-AUC {winner[1].pooled_pr_auc:.3f}."
    )


if __name__ == "__main__":
    main()
