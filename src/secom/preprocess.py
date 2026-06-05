"""Leak-free preprocessing for SECOM.

Everything that learns from data (which columns to drop, imputation
medians, which missing-indicators to add) is bound to ``fit`` so it only
ever sees the training fold. ``transform`` then applies those frozen
decisions to unseen (future) rows.

Encodes hypotheses H3 (missing-as-signal indicators) and the groundwork
for H4 (correlation clustering, provided as a standalone helper).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


class Preprocessor:
    """sklearn-style fit/transform preprocessor with no temporal leakage."""

    def __init__(
        self,
        missing_indicator_min_gap: float = config.MISSING_INDICATOR_MIN_GAP,
        max_missing_indicators: int = config.MAX_MISSING_INDICATORS,
        max_column_missing_frac: float = config.MAX_COLUMN_MISSING_FRAC,
    ) -> None:
        self.missing_indicator_min_gap = missing_indicator_min_gap
        self.max_missing_indicators = max_missing_indicators
        self.max_column_missing_frac = max_column_missing_frac

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Preprocessor":
        # 1. Drop constant columns (no information) and high-missing columns.
        nunique = X.nunique(dropna=True)
        miss_frac = X.isna().mean()
        self.keep_cols_ = [
            c for c in X.columns
            if nunique[c] > 1 and miss_frac[c] <= self.max_column_missing_frac
        ]

        Xk = X[self.keep_cols_]

        # 2. Imputation medians (learned on train only).
        self.medians_ = Xk.median()

        # 3. H3: select columns whose missingness separates the classes.
        is_missing = Xk.isna()
        gap = (is_missing[y == 1].mean() - is_missing[y == 0].mean()).abs()
        self.indicator_cols_ = (
            gap[gap >= self.missing_indicator_min_gap]
            .sort_values(ascending=False)
            .head(self.max_missing_indicators)
            .index.tolist()
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xk = X[self.keep_cols_].copy()

        # Missing-indicator features BEFORE imputation erases the signal.
        indicators = {
            f"{c}__isna": Xk[c].isna().astype(int) for c in self.indicator_cols_
        }

        Xk = Xk.fillna(self.medians_)
        if indicators:
            Xk = pd.concat([Xk, pd.DataFrame(indicators, index=Xk.index)], axis=1)
        return Xk

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def cluster_features(
    X: pd.DataFrame, corr_threshold: float = 0.95
) -> dict[int, list[str]]:
    """Group highly-correlated columns into clusters (H4 groundwork).

    Uses single-linkage agglomeration on (1 - |corr|) distance. Returns a
    mapping cluster_id -> column names. Caller can keep one representative
    per cluster or PCA-reduce each cluster to cut redundancy/variance.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    corr = X.corr().abs().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1.0)
    dist = 1.0 - corr
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)

    labels = fcluster(
        linkage(squareform(dist, checks=False), method="single"),
        t=1.0 - corr_threshold,
        criterion="distance",
    )
    clusters: dict[int, list[str]] = {}
    for col, lab in zip(X.columns, labels):
        clusters.setdefault(int(lab), []).append(col)
    return clusters
