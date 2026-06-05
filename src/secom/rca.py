"""Root-cause helpers for simulated multi-tool analysis."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


TOOL_GROUP_RULES: list[tuple[str, range]] = [
    ("deposition", range(0, 200)),
    ("etch", range(200, 400)),
    ("inspection", range(400, 590)),
]


def _tool_group_for_feature(feature: str) -> str:
    if not feature.startswith("f"):
        return "other"
    try:
        idx = int(feature[1:])
    except Exception:
        return "other"
    for name, r in TOOL_GROUP_RULES:
        if idx in r:
            return name
    return "other"


def map_features_to_tools(feature_names: list[str]) -> dict[str, list[str]]:
    """Map feature names into simulated tool groups."""
    out = defaultdict(list)
    for c in feature_names:
        out[_tool_group_for_feature(c)].append(c)
    # make deterministic order
    for k in out:
        out[k] = sorted(out[k])
    if "other" not in out:
        out["other"] = []
    return dict(out)


def group_attributions(
    shap_values: pd.DataFrame,
    clusters: dict[int, list[str]],
) -> dict[int, float]:
    """Aggregate absolute-SHAP mass by cluster id."""
    if shap_values.empty:
        return {int(cid): 0.0 for cid in clusters}
    abs_mass = shap_values.abs().sum(axis=0)
    out: dict[int, float] = {}
    for cid, cols in clusters.items():
        present = [c for c in cols if c in abs_mass.index]
        out[int(cid)] = float(abs_mass[present].sum()) if present else 0.0
    return out


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute PSI with histogram binning fit on reference sample."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < bins or len(cur) < bins:
        return 0.0

    q = np.quantile(ref, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=q)[0] / len(ref)
    c = np.histogram(cur, bins=q)[0] / len(cur)
    eps = 1e-6
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def explain_late_failure(
    shap_values: pd.DataFrame,
    tool_groups: dict[str, list[str]],
    top_n: int = 5,
):
    """Produce per-tool attribution totals and top features from SHAP rows."""
    if shap_values.empty:
        return {"tool": {}, "features": []}

    abs_mass = shap_values.abs().sum(axis=0).sort_values(ascending=False)
    out = {}
    for tool, cols in tool_groups.items():
        present = [c for c in cols if c in abs_mass.index]
        out[tool] = float(abs_mass[present].sum()) if present else 0.0
    total = sum(out.values()) or 1.0
    confidence = {k: v / total for k, v in out.items()}

    top_feats = abs_mass.head(top_n).index.tolist()
    feature_masses = abs_mass.loc[top_feats].to_dict()
    return {
        "tool": out,
        "tool_confidence": confidence,
        "features": top_feats,
        "feature_masses": feature_masses,
        "summary": (
            sorted(out.items(), key=lambda kv: kv[1], reverse=True)[:3]
        ),
    }
