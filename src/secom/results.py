"""Experiment result persistence helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from . import config

DEFAULT_PATH = config.ARTIFACTS / "results.jsonl"


def log_result(row: dict, path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load_results(path: Path = DEFAULT_PATH) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def render_table(path: Path = DEFAULT_PATH) -> str:
    rows = load_results(path)
    if not rows:
        return "_no results yet_"

    cols = [
        "hypothesis",
        "name",
        "pooled_pr_auc",
        "pooled_roc_auc",
        "mean_recall_at_target",
        "verdict",
    ]

    header = "| " + " | ".join(cols) + " |\n"
    header += "|" + "|".join([" --- " for _ in cols]) + "|\n"
    body = ""
    for row in rows:
        body += (
            "| "
            + " | ".join(
                f"{row[c]:.3f}" if isinstance(row.get(c), float) else str(row.get(c, ""))
                for c in cols
            )
            + " |\n"
        )
    return header + body


def render_summary(path: Path = DEFAULT_PATH) -> str:
    rows = load_results(path)
    if not rows:
        return "No experiments run."

    by_h = {}
    for r in rows:
        by_h.setdefault(r.get("hypothesis", "unknown"), []).append(r)
    lines = ["# SECOM Hypothesis Summary\n"]
    for h in sorted(by_h):
        lines.append(f"\n## {h}")
        for r in by_h[h]:
            rowstr = ", ".join(f"{k}={r.get(k)}" for k in ("name", "pooled_pr_auc", "mean_recall_at_target", "verdict"))
            lines.append(f"- {rowstr}")
    return "\n".join(lines)
