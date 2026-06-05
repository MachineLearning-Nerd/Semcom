"""H4: cluster-reduced features vs all features (paired walk-forward)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from secom.data import load_secom
from secom.experiment import run_experiment
from secom.features import ClusterReducer
from secom.results import log_result
from secom.validation import walk_forward_splits


def reduced_fit_transform(pp, Xtr, ytr):
    Xtr_t = pp.fit_transform(Xtr, ytr)
    red = ClusterReducer(corr_threshold=0.95).fit(Xtr_t)
    Xr = red.transform(Xtr_t)

    def transform(Xte):
        return red.transform(pp.transform(Xte))

    return Xr, transform


def main() -> None:
    d = load_secom()
    splits = list(walk_forward_splits(len(d.y)))
    base = run_experiment(d, splits, model_name="lightgbm", name="H4:all_features")
    red = run_experiment(
        d,
        splits,
        model_name="lightgbm",
        fit_transform=reduced_fit_transform,
        name="H4:clustered",
    )

    print(f"all_features  pooled PR-AUC={base.pooled_pr_auc:.3f}")
    print(f"clustered     pooled PR-AUC={red.pooled_pr_auc:.3f}")
    for tag, res in [("all_features", base), ("clustered", red)]:
        log_result(
            {
                "hypothesis": "H4",
                "name": tag,
                "pooled_pr_auc": res.pooled_pr_auc,
                "pooled_roc_auc": res.pooled_roc_auc,
            }
        )
    print(
        "\nVERDICT: H4 holds if clustered PR-AUC >= all_features (parity with fewer feats = win)."
    )


if __name__ == "__main__":
    main()
