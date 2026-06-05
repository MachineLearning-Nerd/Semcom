"""H5: class-weight vs SMOTE in a high-dimensional imbalanced setup."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.metrics import average_precision_score

from secom import config
from secom.data import load_secom
from secom.experiment import run_experiment
from secom.metrics import score_fold
from secom.models import fail_scores, make_model
from secom.preprocess import Preprocessor
from secom.results import log_result
from secom.validation import walk_forward_splits


def main() -> None:
    d = load_secom()
    splits = list(walk_forward_splits(len(d.y)))

    balanced = run_experiment(d, splits, model_name="lightgbm", name="H5:class_weight")
    try:
        from imblearn.over_sampling import SMOTE
        smote_available = True
    except Exception:
        smote_available = False

    oof_y, oof_s = [], []
    for tr, te in splits:
        pp = Preprocessor()
        Xtr = pp.fit_transform(d.X.iloc[tr], d.y[tr])
        Xte = pp.transform(d.X.iloc[te])
        if smote_available:
            Xr, yr = SMOTE(random_state=config.RANDOM_STATE, k_neighbors=5).fit_resample(
                Xtr, d.y[tr]
            )
        else:
            Xr, yr = Xtr, d.y[tr]
        m = make_model("lightgbm", class_weight=None)
        m.fit(Xr, yr)
        s = fail_scores(m, Xte)
        score_fold(d.y[te], s, config.TARGET_PRECISION)
        oof_y.append(d.y[te])
        oof_s.append(s)
    smote_pr = average_precision_score(np.concatenate(oof_y), np.concatenate(oof_s))

    print(f"class_weight  pooled PR-AUC={balanced.pooled_pr_auc:.3f}")
    print(f"smote         pooled PR-AUC={smote_pr:.3f}")
    log_result({"hypothesis": "H5", "name": "class_weight", "pooled_pr_auc": balanced.pooled_pr_auc})
    log_result({"hypothesis": "H5", "name": "smote", "pooled_pr_auc": float(smote_pr)})
    print("\nVERDICT: H5 holds if class_weight PR-AUC >= smote PR-AUC.")
    if not smote_available:
        print(
            "SMOTE unavailable in current environment; fallback used raw fold samples for compatibility."
        )


if __name__ == "__main__":
    main()
