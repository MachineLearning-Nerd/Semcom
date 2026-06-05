"""H3: missing-indicator features ON vs OFF (paired)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from secom.data import load_secom
from secom.experiment import run_experiment
from secom.preprocess import Preprocessor
from secom.results import log_result
from secom.validation import walk_forward_splits


def main() -> None:
    d = load_secom()
    splits = list(walk_forward_splits(len(d.y)))
    configs = {
        "indicators_off": lambda: Preprocessor(missing_indicator_min_gap=2.0),
        "indicators_on": lambda: Preprocessor(missing_indicator_min_gap=0.05),
    }
    for tag, mk in configs.items():
        res = run_experiment(
            d, splits, model_name="lightgbm", make_preprocessor=mk, name=f"H3:{tag}"
        )
        print(
            f"{tag:16} pooled PR-AUC={res.pooled_pr_auc:.3f}  ROC-AUC={res.pooled_roc_auc:.3f}"
        )
        log_result(
            {
                "hypothesis": "H3",
                "name": tag,
                "pooled_pr_auc": res.pooled_pr_auc,
                "pooled_roc_auc": res.pooled_roc_auc,
            }
        )
    print("\nVERDICT: H3 holds if indicators_on pooled PR-AUC > indicators_off.")


if __name__ == "__main__":
    main()
