"""Load the raw SECOM dataset into aligned (X, y, timestamps).

The raw files are whitespace-delimited with "NaN" string tokens for
missing values. Labels live in a separate file whose rows align 1:1 with
the feature rows and carry a production timestamp -- the temporal axis
that makes honest validation possible.
"""
from __future__ import annotations

from dataclasses import dataclass
import pathlib
from hashlib import sha256

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class SecomData:
    """Immutable container for an aligned, time-sorted SECOM dataset."""

    X: pd.DataFrame          # (n_samples, n_features) raw sensor matrix, NaNs intact
    y: np.ndarray            # (n_samples,) int: 1 = fail (positive), 0 = pass
    timestamps: pd.Series    # (n_samples,) production datetime, sorted ascending

    @property
    def n_fail(self) -> int:
        return int(self.y.sum())

    @property
    def fail_rate(self) -> float:
        return float(self.y.mean())


def _file_sha256(path: pathlib.Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


SECOM_DATASET_URL = "https://archive.ics.uci.edu/dataset/179/secom"
SECOM_MANIFEST = config.DATA_RAW / "manifest.md"


def load_secom(sort_by_time: bool = True) -> SecomData:
    """Read raw SECOM files and return an aligned, time-sorted dataset.

    Args:
        sort_by_time: if True, rows are sorted by production timestamp so
            that positional index order == chronological order (required
            for walk-forward splitting).

    Returns:
        SecomData with NaNs preserved (imputation is a fit-on-train concern).
    """
    X = pd.read_csv(config.SECOM_DATA, sep=r"\s+", header=None, na_values=["NaN"])
    X.columns = [f"f{c}" for c in X.columns]

    labels = pd.read_csv(
        config.SECOM_LABELS,
        sep=r"\s+",
        header=None,
        names=["label", "date", "time"],
        engine="python",  # regex split ignores quotes -> reliably 3 fields
    )
    timestamps = pd.to_datetime(
        labels["date"].str.strip('"') + " " + labels["time"].str.strip('"'),
        format=config.TIMESTAMP_FORMAT,
    )
    y = (labels["label"].to_numpy() == config.RAW_FAIL).astype(int)

    if len(X) != len(y):
        raise ValueError(f"Row mismatch: X={len(X)} labels={len(y)}")

    if sort_by_time:
        order = np.argsort(timestamps.to_numpy(), kind="stable")
        X = X.iloc[order].reset_index(drop=True)
        y = y[order]
        timestamps = timestamps.iloc[order].reset_index(drop=True)

    return SecomData(X=X, y=y, timestamps=timestamps)


def dataset_manifest() -> dict:
    """Emit a small reproducibility manifest for the raw dataset files."""
    return {
        "source": SECOM_DATASET_URL,
        "files": {
            str(config.SECOM_DATA): {
                "exists": config.SECOM_DATA.exists(),
                "sha256": _file_sha256(config.SECOM_DATA) if config.SECOM_DATA.exists() else None,
            },
            str(config.SECOM_LABELS): {
                "exists": config.SECOM_LABELS.exists(),
                "sha256": _file_sha256(config.SECOM_LABELS) if config.SECOM_LABELS.exists() else None,
            },
        },
    }


def write_manifest(path: pathlib.Path | None = None) -> pathlib.Path:
    """Write the dataset manifest to disk for auditability."""
    out_path = path if path is not None else SECOM_MANIFEST
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SECOM raw dataset manifest",
        f"source: {SECOM_DATASET_URL}",
        "",
    ]
    for name, info in dataset_manifest()["files"].items():
        lines.append(f"- {name}")
        lines.append(f"  exists: {info['exists']}")
        lines.append(f"  sha256: {info['sha256']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def export_cleaned_dataset(path: pathlib.Path | None = None) -> pathlib.Path:
    """Write a deterministic cleaned copy of the dataset used by experiments."""
    from .preprocess import Preprocessor

    d = load_secom()
    pp = Preprocessor()
    X = pp.fit_transform(d.X, d.y)
    ts = d.timestamps
    out = X.copy()
    out["timestamp"] = ts
    out["label"] = d.y
    out_path = path if path is not None else config.ARTIFACTS / "cleaned_secom.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    return out_path
