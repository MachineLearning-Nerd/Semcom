"""H15: produce dashboard-ready artifacts (charts + exported result snapshot).

This is a lightweight reporting task to support proposal-ready visuals.
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from secom.data import load_secom, write_manifest
from secom.results import load_results
from secom.visuals import plot_class_distribution, plot_precision_recall_like, plot_feature_attribution, plot_tool_heatmap


def main() -> None:
    d = load_secom()
    rows = load_results()

    if not rows:
        print("No results to render. Run exp_h*.py first.")
        return

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_class_distribution(d.y, out_dir / "class_distribution.png")
    plot_precision_recall_like(rows, out_dir / "precision_recall_tradeoff.png")

    # Build a simple tool-attribution artifact if available.
    tool_rows = {}
    tool_path = out_dir / "h14_late_defect_rca.json"
    if tool_path.exists():
        try:
            payload = json.loads(tool_path.read_text(encoding="utf-8"))
            tool_rows = {
                name: float(value)
                for name, value in payload.get("top_tool_votes", [])
                if value is not None
            }
            if tool_rows:
                plot_tool_heatmap(tool_rows, out_dir / "tool_attribution.png")
        except Exception:
            pass

    # Build feature attribution from H8 CSV if present.
    h8_csv = out_dir / "h8_cluster_attribution.csv"
    if h8_csv.exists():
        df = pd.read_csv(h8_csv)
        if {"cluster_id", "mass"}.issubset(df.columns):
            feats = {
                f"cluster_{int(c)}": float(m)
                for c, m in list(zip(df["cluster_id"], df["mass"]))[:15]
            }
            plot_feature_attribution(feats, out_dir / "feature_attribution_summary.png", top_n=15)

    # snapshot for investor-friendly review
    manifest = write_manifest(out_dir / "h15_dataset_manifest.txt")
    summary = {
        "generated_at": datetime.now().isoformat(),
        "n_rows": int(len(d.y)),
        "n_features": int(d.X.shape[1]),
        "fail_rate": float(d.fail_rate),
        "n_results_rows": int(len(rows)),
        "artifact_manifest": str(manifest),
    }
    (out_dir / "h15_dashboard_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("H15 dashboard artifacts written under artifacts/")


if __name__ == "__main__":
    main()
