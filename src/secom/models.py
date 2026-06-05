"""Model factory + uniform failure-score interface.

All models here expose score direction as "higher score = more likely FAIL".
This lets the experiment runner stay model-agnostic and avoids ad-hoc extraction
logic in each hypothesis script.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import config

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency in minimal env
    LGBMClassifier = None

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def make_model(name: str, **overrides: Any):
    """Create a model by short name.

    Supported names: ``logreg``, ``rf``, ``lightgbm``, ``iforest``.
    Extra kwargs are passed through to the concrete estimator.
    """
    name = name.lower()
    if name == "logreg":
        params = dict(
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            C=1.0,
            random_state=config.RANDOM_STATE,
        )
        return LogisticRegression(**{**params, **overrides})

    if name == "rf":
        params = dict(
            n_estimators=400,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )
        return RandomForestClassifier(**{**params, **overrides})

    if name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError(
                "lightgbm is not installed. Install with `uv pip install lightgbm`."
            )
        params = dict(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
            verbose=-1,
        )
        return LGBMClassifier(**{**params, **overrides})

    if name == "iforest":
        params = dict(
            n_estimators=300,
            contamination="auto",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        return IsolationForest(**{**params, **overrides})

    raise ValueError(f"unknown model: {name}")


def fail_scores(model, X) -> np.ndarray:
    """Return failure likelihood score for each row.

    Uses:
    - predict_proba (if available): use class 1/FAIL probability.
    - score_samples (IsolationForest): negate so higher => more anomalous.
    - decision_function fallback.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "score_samples"):
        return -np.asarray(model.score_samples(X))
    return np.asarray(model.decision_function(X))
