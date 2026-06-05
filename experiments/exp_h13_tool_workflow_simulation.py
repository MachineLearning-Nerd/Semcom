"""H13: simulated multi-tool workflow with staged score signatures.

Assume a serial process:
Deposition -> Etch -> Inspection.
For each fold, train cumulative stage models and emit stage-level AUCs.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from secom.data import load_secom
from secom.metrics import score_fold
from secom.models import fail_scores, make_model
from secom.preprocess import Preprocessor
from secom.rca import map_features_to_tools
from secom.results import log_result
from secom.validation import walk_forward_splits


def _collect_stage_metrics(y: np.ndarray, step_scores: dict[str, np.ndarray]):
    stage_pr = {}
    stage_rec = {}
    for step, s in step_scores.items():
        fs = score_fold(y, s, 0.25)
        stage_pr[step] = fs.pr_auc
        stage_rec[step] = fs.recall_at_target
    return stage_pr, stage_rec


def main() -> None:
    d = load_secom()
    splits = list(walk_forward_splits(len(d.y)))
    step_order = ["deposition", "etch", "inspection"]

    oof = {step: [] for step in step_order}
    y_all = []
    late_narratives = []

    for tr, te in splits:
        pp = Preprocessor()
        Xtr = pp.fit_transform(d.X.iloc[tr], d.y[tr])
        Xte = pp.transform(d.X.iloc[te])
        yte = d.y[te]

        groups = map_features_to_tools(list(Xtr.columns))
        cumulative = []
        step_scores = {}

        for step in step_order:
            cols = [c for c in groups.get(step, []) if c in Xtr.columns]
            if not cols:
                continue
            cumulative.extend(cols)
            # deterministic order while avoiding duplicates
            cumulative_unique = list(dict.fromkeys(cumulative))
            m = make_model("lightgbm")
            m.fit(Xtr[cumulative_unique], ytr := d.y[tr])
            s = fail_scores(m, Xte[cumulative_unique])
            step_scores[step] = s

            oof[step].append(s)

        if "inspection" in step_scores and len(yte) > 0:
            # choose one late-stage suspect list from the inspection step
            top_idx = np.argsort(step_scores["inspection"])[::-1][: min(3, len(step_scores["inspection"]))]
            for local_pos in top_idx[:3]:
                if yte[local_pos] == 1:
                    break
                fail_idx = local_pos
            # Build one narrative for the highest inspection score only.
            if len(top_idx):
                fail_idx = int(top_idx[0])
                row = {step: float(step_scores[step][fail_idx]) for step in step_order if step in step_scores}
                suspected = max(row, key=row.get)
                late_narratives.append(
                    {
                        "stage": suspected,
                        "step_scores": row,
                        "inspection_score": row.get("inspection", 0.0),
                    }
                )

        y_all.append(yte)

    y_all = np.concatenate(y_all)
    for step in step_order:
        if not oof[step]:
            continue
        s = np.concatenate(oof[step])
        fs = score_fold(y_all, s, 0.25)
        log_result(
            {
                "hypothesis": "H13",
                "name": f"stage_{step}",
                "pooled_pr_auc": fs.pr_auc,
                "pooled_roc_auc": fs.roc_auc,
                "mean_recall_at_target": fs.recall_at_target,
            }
        )
        print(f"{step:12} PR-AUC={fs.pr_auc:.3f} Recall@25={fs.recall_at_target:.3f}")

    # Persist stage signatures for dashboard/inspection.
    Path("artifacts").mkdir(exist_ok=True, parents=True)
    Path("artifacts/h13_late_narratives.json").write_text(
        pd.Series(late_narratives).to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    # quick visual to mirror workflow behavior
    stage_lines = []
    names = []
    for step in step_order:
        scores = np.concatenate(oof[step]) if oof[step] else np.array([])
        if len(scores):
            names.append(step)
            stage_lines.append(np.mean(scores))
    if names:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(names, stage_lines, marker="o")
        ax.set_title("H13 staged mean risk score")
        ax.set_xlabel("Tool stage")
        ax.set_ylabel("mean score")
        fig.tight_layout()
        fig.savefig("artifacts/h13_stage_signatures.png", dpi=140)
        plt.close(fig)

    # log one summary row with narrative count
    log_result(
        {
            "hypothesis": "H13",
            "name": "narratives",
            "pooled_pr_auc": 0.0,
            "pooled_roc_auc": 0.0,
            "mean_recall_at_target": 0.0,
            "n_cases": len(late_narratives),
        }
    )

    print(f"\nVERDICT: H13 supports staged workflow if inspection-only stage outperforms earlier stages for late defects.")


if __name__ == "__main__":
    main()
