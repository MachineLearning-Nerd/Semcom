"""H14: trace late-stage defects back to upstream tool clusters.

Use walk-forward models and per-detection explanations to provide an explicit
"suspected upstream" summary for inspection-stage high-risk failures.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path

import numpy as np
import pandas as pd

from secom.data import load_secom
from secom.metrics import score_fold
from secom.models import fail_scores, make_model
from secom.preprocess import Preprocessor
from secom.rca import explain_late_failure, map_features_to_tools
from secom.results import log_result
from secom.validation import walk_forward_splits


# deterministic fallback when SHAP isn't installed or too slow

def _fallback_attributions(Xte: pd.DataFrame, yte: np.ndarray, model) -> pd.DataFrame:
    imp = np.abs(getattr(model, "feature_importances_", np.ones(Xte.shape[1])))
    if len(imp) != Xte.shape[1]:
        imp = np.ones(Xte.shape[1])
    return pd.DataFrame(np.repeat([imp], len(Xte), axis=0), columns=Xte.columns)


def _shap_values_safe(model, Xte: pd.DataFrame) -> pd.DataFrame:
    try:
        import shap
    except Exception:
        raise RuntimeError("SHAP not available")

    expl = shap.TreeExplainer(model)
    raw = expl.shap_values(Xte)
    if isinstance(raw, list):
        raw = raw[1]
    return pd.DataFrame(np.asarray(raw), columns=Xte.columns)


def main() -> None:
    d = load_secom()
    splits = list(walk_forward_splits(len(d.y)))

    late_rows = []
    tool_votes = Counter()
    all_feature_votes = Counter()
    y_true = []
    y_scores = []

    for tr, te in splits:
        pp = Preprocessor()
        Xtr = pp.fit_transform(d.X.iloc[tr], d.y[tr])
        Xte = pp.transform(d.X.iloc[te])
        ytr = d.y[tr]
        yte = d.y[te]

        m = make_model("lightgbm")
        m.fit(Xtr, ytr)
        scores = fail_scores(m, Xte)
        y_true.append(yte)
        y_scores.append(scores)

        try:
            shap_df = _shap_values_safe(m, Xte)
        except Exception:
            shap_df = _fallback_attributions(Xte, yte, m).abs()

        tool_map = map_features_to_tools(list(shap_df.columns))
        summary = explain_late_failure(shap_df, tool_map, top_n=6)
        late_mask = yte == 1
        late_idx = np.where(late_mask)[0]
        if len(late_idx) == 0:
            continue

        top_fail = late_idx[np.argsort(scores[late_mask])[::-1][:3]]
        for li in top_fail:
            global_rank = int(np.where(np.argsort(scores)[::-1] == li)[0][0])
            row = {
                "inspection_step_rank": int(global_rank) + 1,
                "inspection_score": float(scores[li]),
                "top_tools": [t for t, _ in summary["summary"]],
                "top_features": list(summary["features"]),
                "tool_confidence": summary.get("tool_confidence", {}),
            }
            late_rows.append(row)
            if summary["summary"]:
                tool_votes[summary["summary"][0][0]] += 1
            for feat in summary["features"]:
                all_feature_votes[feat] += 1

    if y_true:
        y_true = np.concatenate(y_true)
        y_scores = np.concatenate(y_scores)
        fold = score_fold(y_true, y_scores, 0.25)
    else:
        fold = None

    out = Path("artifacts/h14_late_defect_rca.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "n_cases": len(late_rows),
                "top_tool_votes": tool_votes.most_common(),
                "top_feature_votes": all_feature_votes.most_common(10),
                "late_cases": late_rows[:20],
                "pooled_pr_auc": float(fold.pr_auc) if fold else 0.0,
                "pooled_recall_at25": float(fold.recall_at_target) if fold else 0.0,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    log_result(
        {
            "hypothesis": "H14",
            "name": "late_defect_attribution",
            "pooled_pr_auc": float(fold.pr_auc) if fold else 0.0,
            "mean_recall_at_target": float(fold.recall_at_target) if fold else 0.0,
            "pooled_roc_auc": float(fold.roc_auc) if fold else 0.0,
            "n_cases": len(late_rows),
            "top_tool": tool_votes.most_common(1)[0][0] if tool_votes else "",
        }
    )

    print("VERDICT: H14 holds if late failures repeatedly map to coherent upstream tool drivers.")
    print(f"Top driver tool: {tool_votes.most_common(1)}")


if __name__ == "__main__":
    main()
