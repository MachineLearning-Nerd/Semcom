"""H8: fail-predictive signal concentration in a small set of clusters.

Prefer SHAP attributions when the dependency is installed; otherwise fall back
to model feature-importance mass so this experiment still runs in reduced envs.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from secom.data import load_secom
from secom.experiment import run_experiment
from secom.metrics import score_fold
from secom.models import make_model, fail_scores
from secom.preprocess import Preprocessor, cluster_features
from secom.rca import group_attributions
from secom.results import log_result
from secom.validation import walk_forward_splits


def _safe_shap_values(model, X: pd.DataFrame) -> pd.DataFrame:
    try:
        import shap
    except Exception:
        raise RuntimeError("SHAP unavailable")

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        raw = raw[1]
    if len(np.shape(raw)) == 2:
        return pd.DataFrame(np.asarray(raw), columns=X.columns, index=X.index)
    if np.ndim(raw) == 3:
        return pd.DataFrame(np.asarray(raw)[..., 1], columns=X.columns, index=X.index)
    raise RuntimeError("Unexpected SHAP output shape")


def main() -> None:
    d = load_secom()
    splits = list(walk_forward_splits(len(d.y)))

    all_attrib = []
    fold_lines = []

    for split_id, (tr, te) in enumerate(splits, 1):
        pp = Preprocessor(missing_indicator_min_gap=2.0)
        Xtr = pp.fit_transform(d.X.iloc[tr], d.y[tr])
        Xte = pp.transform(d.X.iloc[te])
        yte = d.y[te]

        m = make_model("lightgbm")
        m.fit(Xtr, d.y[tr])
        s = fail_scores(m, Xte)
        fold_lines.append(score_fold(yte, s, 0.25))

        try:
            sv = _safe_shap_values(m, Xte)
        except Exception:
            # SHAP fallback: use model importances duplicated across rows.
            imp = getattr(m, "feature_importances_")
            sv = pd.DataFrame(
                np.repeat(np.asarray([np.abs(imp)], dtype=float), repeats=len(Xte), axis=0),
                columns=Xte.columns,
                index=Xte.index,
            )

        all_attrib.append(sv)

    attrib = pd.concat(all_attrib, axis=0)
    clusters = cluster_features(attrib)
    cluster_mass = group_attributions(attrib, clusters)
    sorted_cluster = sorted(cluster_mass.items(), key=lambda kv: kv[1], reverse=True)
    top3 = sorted_cluster[:3]

    total = sum(cluster_mass.values()) or 1.0
    top3_share = sum(v for _, v in top3) / total
    top_cluster_ids = [cid for cid, _ in top3]
    print(f"Top clusters: {top_cluster_ids}")
    print(f"Top-3 concentration share: {top3_share:.3f}")

    out_path = Path("artifacts/h8_cluster_attribution.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "cluster_id": [int(cid) for cid, _ in sorted_cluster],
            "mass": [float(v) for _, v in sorted_cluster],
            "share": [float(v) / total for _, v in sorted_cluster],
        }
    ).to_csv(out_path, index=False)

    # Also compare against a minimal fold-based baseline for continuity.
    res = run_experiment(d, splits, model_name="lightgbm", name="H8:baseline")
    log_result(
        {
            "hypothesis": "H8",
            "name": "cluster_concentration",
            "pooled_pr_auc": res.pooled_pr_auc,
            "pooled_roc_auc": res.pooled_roc_auc,
            "mean_recall_at_target": res.mean_recall_at_target,
            "top3_cluster_share": float(top3_share),
            "top3_cluster_ids": ",".join(map(str, top_cluster_ids)),
        }
    )

    print(
        "\nVERDICT: H8 holds if concentrated signal appears in a small number"
        " of clusters and top3_share is materially above 0.33."
    )


if __name__ == "__main__":
    main()
