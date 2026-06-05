"""H12: combine heterogeneous models and compare PR-AUC/recall/precision."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from secom.data import load_secom
from secom.metrics import score_fold
from secom.models import fail_scores, make_model
from secom.preprocess import Preprocessor
from secom.results import log_result
from secom.validation import walk_forward_splits


def _evaluate_fold(X, y, tr, te, seeds=(42,)):
    Xtr = X.iloc[tr]
    ytr = y[tr]
    Xte = X.iloc[te]
    yte = y[te]

    pp = Preprocessor(missing_indicator_min_gap=2.0)
    Xtr = pp.fit_transform(Xtr, ytr)
    Xte = pp.transform(Xte)

    log_scores = []
    lgbm_scores = []
    for seed in seeds:
        m1 = make_model("logreg", random_state=seed)
        m1.fit(Xtr, ytr)
        log_scores.append(fail_scores(m1, Xte))

        m2 = make_model("lightgbm", random_state=seed)
        m2.fit(Xtr, ytr)
        lgbm_scores.append(fail_scores(m2, Xte))

    log_scores = np.mean(np.asarray(log_scores), axis=0)
    lgbm_scores = np.mean(np.asarray(lgbm_scores), axis=0)
    blend = (log_scores + lgbm_scores) / 2.0
    return (log_scores, lgbm_scores, blend, yte)


def main() -> None:
    d = load_secom()
    splits = list(walk_forward_splits(len(d.y)))

    fold_parts = {
        "logreg": [],
        "lgbm": [],
        "blend": [],
    }
    y_parts = []

    for tr, te in splits:
        # Use a single seed for faster wall-clock execution while preserving the
        # heterogeneous-baseline comparison between logistic and LightGBM.
        log_s, lgb_s, blend_s, yte = _evaluate_fold(d.X, d.y, tr, te, seeds=(0,))
        fold_parts["logreg"].append(log_s)
        fold_parts["lgbm"].append(lgb_s)
        fold_parts["blend"].append(blend_s)
        y_parts.append(yte)

    y_all = np.concatenate(y_parts)
    y_true = y_all
    out_rows = {}
    for key in fold_parts:
        scores = np.concatenate(fold_parts[key])
        fold_lines = score_fold(y_true, scores, 0.25)
        out_rows[key] = {
            "pooled_pr_auc": float(np.mean(scores if len(scores) else 0.0)),
            "pooled_roc_auc": fold_lines.roc_auc,
            "mean_recall_at_target": fold_lines.recall_at_target,
            "roc_auc": fold_lines.roc_auc,
        }

    # recompute PR-AUC correctly
    from sklearn.metrics import average_precision_score, roc_auc_score

    for key in fold_parts:
        scores = np.concatenate(fold_parts[key])
        pr = float(average_precision_score(y_true, scores)) if len(scores) else 0.0
        roc = float(roc_auc_score(y_true, scores)) if len(scores) else 0.0
        out_rows[key]["pooled_pr_auc"] = pr
        out_rows[key]["pooled_roc_auc"] = roc

    print(f"logreg pooled PR-AUC={out_rows['logreg']['pooled_pr_auc']:.3f}")
    print(f"lgbm  pooled PR-AUC={out_rows['lgbm']['pooled_pr_auc']:.3f}")
    print(f"blend pooled PR-AUC={out_rows['blend']['pooled_pr_auc']:.3f}")

    for name, vals in out_rows.items():
        log_result(
            {
                "hypothesis": "H12",
                "name": name,
                "pooled_pr_auc": vals["pooled_pr_auc"],
                "pooled_roc_auc": vals["pooled_roc_auc"],
                "mean_recall_at_target": vals["mean_recall_at_target"],
            }
        )

    best = max(out_rows, key=lambda k: out_rows[k]["pooled_pr_auc"])
    print(
        f"\nVERDICT: H12 holds if blend PR-AUC > max(single models) with best={best}."
    )


if __name__ == "__main__":
    main()
