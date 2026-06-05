"""H2: time-held-out threshold tuning vs default 0.5 cutoff."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import precision_score, recall_score

from secom import config
from secom.data import load_secom
from secom.experiment import run_experiment
from secom.metrics import recall_at_precision
from secom.results import log_result
from secom.validation import walk_forward_splits


def main() -> None:
    d = load_secom()
    splits = list(walk_forward_splits(len(d.y)))
    res = run_experiment(d, splits, model_name="lightgbm", name="H2")

    default_pred = (res.oof_scores >= 0.5).astype(int)
    default_recall = recall_score(res.oof_y, default_pred, zero_division=0)
    default_prec = precision_score(res.oof_y, default_pred, zero_division=0)

    tuned_recall, thr = recall_at_precision(
        res.oof_y, res.oof_scores, config.TARGET_PRECISION
    )
    print(f"default@0.5 : recall={default_recall:.3f} precision={default_prec:.3f}")
    print(f"tuned@>={int(config.TARGET_PRECISION*100)}% : recall={tuned_recall:.3f} threshold={thr:.4f}")
    log_result(
        {
            "hypothesis": "H2",
            "name": "default_0.5",
            "mean_recall_at_target": default_recall,
            "pooled_pr_auc": res.pooled_pr_auc,
        }
    )
    log_result(
        {
            "hypothesis": "H2",
            "name": f"tuned_{int(config.TARGET_PRECISION*100)}p",
            "mean_recall_at_target": tuned_recall,
            "pooled_pr_auc": res.pooled_pr_auc,
        }
    )
    print("\nVERDICT: H2 holds if tuned recall >> default recall at usable precision.")


if __name__ == "__main__":
    main()
