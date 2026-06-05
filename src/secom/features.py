"""Feature engineering transformers for walk-forward experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif

from .preprocess import cluster_features


@dataclass
class UnivariateScreen(BaseEstimator, TransformerMixin):
    """Leak-safe univariate selector fit on training folds only."""

    k: int = 40

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")
        Xf = X.dropna(axis=1, how="all")
        if Xf.empty:
            self._selector = None
            return self
        # fill infinities so stats can compute, preserve ranking shape
        Xnum = Xf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self._selector = SelectKBest(score_func=f_classif, k=min(self.k, Xnum.shape[1]))
        self._selector.fit(Xnum, y)
        self.columns_ = Xnum.columns
        self._features = list(self.columns_[self._selector.get_support()].tolist())
        self.selected_ = list(self._features)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "_features"):
            return X.copy()
        keep = [c for c in self._features if c in X.columns]
        return X[keep].copy()

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


@dataclass
class ClusterReducer(BaseEstimator, TransformerMixin):
    """Collapse correlated feature clusters into representative mean features."""

    corr_threshold: float = 0.95
    method: str = "mean"

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        self.clusters_ = cluster_features(X, corr_threshold=self.corr_threshold)
        # Sort cluster ids for deterministic output
        self.cluster_ids_ = sorted(self.clusters_)
        self._all_cluster_cols = set(
            col for cols in self.clusters_.values() for col in cols
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "clusters_"):
            return X.copy()
        out = {}
        for cid in self.cluster_ids_:
            cols = [c for c in self.clusters_[cid] if c in X.columns]
            if not cols:
                continue
            if self.method == "mean":
                out[f"cluster_{cid}_mean"] = X[cols].mean(axis=1)
            elif self.method == "sum":
                out[f"cluster_{cid}_sum"] = X[cols].sum(axis=1)
            else:
                raise ValueError("method must be 'mean' or 'sum'")
        for c in self.columns_:
            if c not in self._all_cluster_cols:
                out[c] = X[c]
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
