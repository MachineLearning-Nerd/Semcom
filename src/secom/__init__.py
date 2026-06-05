"""SECOM failure-prediction PoC.

Two tracks (per design decision): a time-validated failure *detector*
and a root-cause-analysis (RCA) story, both evaluated walk-forward to
respect the dataset's strong temporal drift.
"""
__all__ = [
    "config",
    "data",
    "preprocess",
    "metrics",
    "validation",
    "experiment",
    "features",
    "models",
    "results",
    "rca",
    "visuals",
]
