"""Dashboard-ready plotting helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(y, path: Path) -> None:
    path = Path(path)
    _ensure_parent(path)
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    labels = ["pass", "fail"]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, counts)
    ax.set_title("SECOM class distribution")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_precision_recall_like(rows: list[dict], path: Path) -> None:
    path = Path(path)
    _ensure_parent(path)
    ys = [float(r.get("mean_recall_at_target", 0.0)) for r in rows]
    xs = [float(r.get("pooled_pr_auc", 0.0)) for r in rows]
    names = [f"{r.get('hypothesis','H')}:{r.get('name','run')}" for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(xs, ys)
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, (x, y), fontsize=7)
    ax.set_xlabel("pooled PR-AUC")
    ax.set_ylabel(f"recall@target")
    ax.set_title("Model tradeoffs")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_tool_heatmap(group_importance: dict[str, float], path: Path) -> None:
    path = Path(path)
    _ensure_parent(path)
    names = list(group_importance.keys())
    vals = np.array(list(group_importance.values()), dtype=float)
    fig, ax = plt.subplots(figsize=(6, 1.8 + 0.2 * len(names)))
    im = ax.imshow(vals[None, :], aspect="auto", cmap="viridis")
    ax.set_yticks([0])
    ax.set_yticklabels(["attribution"])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20)
    fig.colorbar(im, ax=ax, orientation="vertical")
    ax.set_title("Per-tool attribution (relative)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_feature_attribution(
    feature_scores: dict[str, float] | list[tuple[str, float]],
    path: Path,
    top_n: int = 15,
) -> None:
    path = Path(path)
    _ensure_parent(path)
    items = (
        feature_scores.items()
        if isinstance(feature_scores, dict)
        else feature_scores
    )
    items = sorted(items, key=lambda kv: float(kv[1]), reverse=True)[:top_n]
    if not items:
        return

    names = [k for k, _ in items]
    vals = np.array([float(v) for _, v in items], dtype=float)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(names))))
    y = np.arange(len(names))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("attribution mass")
    ax.set_title("Top feature attributions")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_drift_curves(
    psi: list[float],
    fail_rates: list[float],
    path: Path,
) -> None:
    path = Path(path)
    _ensure_parent(path)
    x = np.arange(len(psi))
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(x, psi, marker="o", label="PSI drift")
    ax1.set_xlabel("window")
    ax1.set_ylabel("mean PSI", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(x, fail_rates, marker="s", color="#ff7f0e", label="fail rate")
    ax2.set_ylabel("failure rate", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    ax1.set_title("Population stability index vs observed fail-rate")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
