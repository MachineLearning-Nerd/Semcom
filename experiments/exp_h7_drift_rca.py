"""H7: early-window vs late-window failure drivers differ (drift RCA).

Compare the top predictive features from the first half of the timeline against
those from the second half. A low Jaccard overlap suggests process drift.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from secom.preprocess import Preprocessor
from secom.results import log_result


def _top_features(X: pd.DataFrame, y: np.ndarray, top_k: int = 30) -> list[str]:
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if X.empty:
        return []
    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0)
    k = min(top_k, X.shape[1])
    idx = np.argsort(scores)[::-1][:k]
    return X.columns[idx].tolist()


def main() -> None:
    data_path = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "secom.data"
    if not data_path.exists():
        raise FileNotFoundError(f"Expected raw dataset at {data_path}")

    # Import lazily to avoid pulling heavy deps during local smoke tests.
    from secom.data import load_secom

    d = load_secom()
    n = len(d.y)
    cut = n // 2

    p = Preprocessor(missing_indicator_min_gap=2.0)
    X0 = p.fit_transform(d.X.iloc[:cut], d.y[:cut])
    f0 = _top_features(X0, d.y[:cut])

    p2 = Preprocessor(missing_indicator_min_gap=2.0)
    X1 = p2.fit_transform(d.X.iloc[cut:], d.y[cut:])
    f1 = _top_features(X1, d.y[cut:])

    set0, set1 = set(f0), set(f1)
    overlap = len(set0 & set1) / len(set0 | set1) if (set0 | set1) else 0.0

    print(f"Early top features: {', '.join(f0[:8])}")
    print(f"Late top features: {', '.join(f1[:8])}")
    print(f"Jaccard overlap (top-30): {overlap:.3f}")

    log_result(
        {
            "hypothesis": "H7",
            "name": "driver_overlap",
            "jaccard": float(overlap),
            "pooled_pr_auc": 0.0,
        }
    )

    if overlap < 0.35:
        print("VERDICT: H7 holds if overlap is low; the top drivers drift across time.")
    else:
        print("VERDICT: H7 does not show strong driver drift under this setup.")


if __name__ == "__main__":
    main()
