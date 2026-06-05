"""H1: quantify the random-CV vs walk-forward gap."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from secom.data import load_secom
from secom.experiment import run_experiment
from secom.results import log_result
from secom.validation import walk_forward_splits, random_stratified_splits


def main() -> None:
    d = load_secom()
    wf = list(walk_forward_splits(len(d.y)))
    rnd = random_stratified_splits(d.y)

    for tag, splits in [("walk_forward", wf), ("random_kfold", rnd)]:
        res = run_experiment(d, splits, model_name="logreg", name=f"H1:{tag}")
        print(
            f"{tag:14} pooled PR-AUC={res.pooled_pr_auc:.3f}  ROC-AUC={res.pooled_roc_auc:.3f}  "
            f"recall@{int(0.25*100)}p={res.mean_recall_at_target:.3f}"
        )
        log_result(
            {
                "hypothesis": "H1",
                "name": tag,
                "pooled_pr_auc": res.pooled_pr_auc,
                "pooled_roc_auc": res.pooled_roc_auc,
                "mean_recall_at_target": res.mean_recall_at_target,
            }
        )

    print(
        "\nVERDICT: H1 holds if random_kfold pooled PR-AUC > walk_forward "
        "(optimism gap). Walk-forward is the number you can trust."
    )


if __name__ == "__main__":
    main()
