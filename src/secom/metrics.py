"""Evaluation metrics tuned for rare-event detection.

Accuracy is intentionally absent: predicting "all pass" scores 93.4% here.
We center on PR-AUC and recall at a fixed precision floor -- the metrics
that reflect catching failures without drowning operators in false alarms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass(frozen=True)
class FoldScore:
    roc_auc: float
    pr_auc: float            # average precision = area under PR curve
    recall_at_target: float  # recall achievable at >= target precision
    threshold: float         # score cutoff achieving that operating point
    n_test: int
    n_fail_test: int


def recall_at_precision(
    y_true: np.ndarray, scores: np.ndarray, target_precision: float
) -> tuple[float, float]:
    """Best recall achievable while precision >= target_precision.

    Returns (recall, threshold). If no threshold reaches the precision
    floor, returns (0.0, +inf) -- i.e. "cannot operate this safely".
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns len(thresholds)+1 points; align by
    # dropping the final (recall=0, precision=1) sentinel.
    precision, recall = precision[:-1], recall[:-1]
    ok = precision >= target_precision
    if not ok.any():
        return 0.0, float("inf")
    best = np.argmax(recall * ok)
    return float(recall[best]), float(thresholds[best])


def score_fold(
    y_true: np.ndarray, scores: np.ndarray, target_precision: float
) -> FoldScore:
    rec, thr = recall_at_precision(y_true, scores, target_precision)
    return FoldScore(
        roc_auc=float(roc_auc_score(y_true, scores)),
        pr_auc=float(average_precision_score(y_true, scores)),
        recall_at_target=rec,
        threshold=thr,
        n_test=len(y_true),
        n_fail_test=int(y_true.sum()),
    )
