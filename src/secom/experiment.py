"""Reusable walk-forward experiment runner for all hypotheses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from . import config
from .data import SecomData
from .metrics import FoldScore, score_fold
from .models import fail_scores, make_model
from .preprocess import Preprocessor


@dataclass
class ExperimentResult:
    name: str
    fold_scores: list[FoldScore]
    oof_y: np.ndarray
    oof_scores: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def mean_roc_auc(self) -> float:
        return float(np.mean([f.roc_auc for f in self.fold_scores])) if self.fold_scores else 0.0

    @property
    def std_roc_auc(self) -> float:
        return float(np.std([f.roc_auc for f in self.fold_scores])) if self.fold_scores else 0.0

    @property
    def mean_pr_auc(self) -> float:
        return float(np.mean([f.pr_auc for f in self.fold_scores])) if self.fold_scores else 0.0

    @property
    def std_pr_auc(self) -> float:
        return float(np.std([f.pr_auc for f in self.fold_scores])) if self.fold_scores else 0.0

    @property
    def mean_recall_at_target(self) -> float:
        return (
            float(np.mean([f.recall_at_target for f in self.fold_scores]))
            if self.fold_scores
            else 0.0
        )

    @property
    def pooled_pr_auc(self) -> float:
        if len(self.oof_y) == 0:
            return 0.0
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(self.oof_y, self.oof_scores))

    @property
    def pooled_roc_auc(self) -> float:
        if len(self.oof_y) == 0:
            return 0.0
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(self.oof_y, self.oof_scores))


def run_experiment(
    data: SecomData,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    model_name: str = "logreg",
    make_preprocessor: Callable[[], Preprocessor] = Preprocessor,
    target_precision: float = config.TARGET_PRECISION,
    seeds: tuple[int, ...] = (config.RANDOM_STATE,),
    name: str = "experiment",
    fit_transform: callable | None = None,
) -> ExperimentResult:
    """Run a configuration over provided folds.

    For each fold:
    - preprocess fit on training rows only
    - train one or more seeded models
    - average seeded scores
    - score held-out rows with `fail_scores`

    Returns per-fold and OOF aggregates required by all downstream tasks.
    """

    fold_scores: list[FoldScore] = []
    y_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []

    for train_idx, test_idx in splits:
        Xtr_raw = data.X.iloc[train_idx]
        ytr = data.y[train_idx]
        Xte_raw = data.X.iloc[test_idx]
        yte = data.y[test_idx]

        pp = make_preprocessor()
        if fit_transform is None:
            Xtr = pp.fit_transform(Xtr_raw, ytr)
            Xte = pp.transform(Xte_raw)
        else:
            Xtr, xform = fit_transform(pp, Xtr_raw, ytr)
            Xte = xform(Xte_raw)

        seeded_scores: list[np.ndarray] = []
        for seed in seeds:
            model = make_model(model_name, random_state=seed)
            if model_name == "iforest":
                model.fit(Xtr)
            else:
                model.fit(Xtr, ytr)
            seeded_scores.append(fail_scores(model, Xte))

        scores = np.mean(np.asarray(seeded_scores), axis=0)
        fold_scores.append(score_fold(yte, scores, target_precision=target_precision))
        y_parts.append(yte)
        score_parts.append(scores)

    return ExperimentResult(
        name=name,
        fold_scores=fold_scores,
        oof_y=np.concatenate(y_parts) if y_parts else np.array([]),
        oof_scores=np.concatenate(score_parts) if score_parts else np.array([]),
        meta={"model_name": model_name, "n_folds": len(fold_scores)},
    )
