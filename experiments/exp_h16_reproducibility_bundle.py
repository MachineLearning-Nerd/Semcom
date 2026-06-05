"""H16: collect reproducibility metadata and execution log for reviewers."""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime
from pathlib import Path

import platform
import sys as _sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from secom import config
from secom.data import dataset_manifest, write_manifest
from secom.results import load_results, render_summary


def main() -> None:
    out = Path("artifacts/h16_reproducibility.json")
    out.parent.mkdir(exist_ok=True, parents=True)

    py = {
        "python": _sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
        "numpy_version": np.__version__,
        "random_state": config.RANDOM_STATE,
        "n_splits": config.N_SPLITS,
        "min_train_frac": config.MIN_TRAIN_FRAC,
        "embargo": config.EMBARGO,
    }
    manifest = dataset_manifest()
    write_manifest(Path("data/raw/manifest.md"))

    results = load_results()
    summary = render_summary()

    payload = {
        "environment": py,
        "dataset": manifest,
        "experiment_count": len(results),
        "hypotheses_recorded": sorted({r.get("hypothesis", "") for r in results}),
        "result_summary_md": summary,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote reproducibility bundle to {out}")


if __name__ == "__main__":
    main()
