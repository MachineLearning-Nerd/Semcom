"""Central configuration: paths, constants, and tunables.

Keeping these in one place makes experiments reproducible and avoids
magic numbers scattered across modules.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
SECOM_DATA = DATA_RAW / "secom.data"
SECOM_LABELS = DATA_RAW / "secom_labels.data"
ARTIFACTS = PROJECT_ROOT / "artifacts"

# --- Labels ------------------------------------------------------------------
# The raw file encodes pass = -1, fail = +1. We map fail -> 1 (positive class)
# because failure is the event of interest (the minority we want recall on).
RAW_PASS = -1
RAW_FAIL = 1
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"  # e.g. "19/07/2008 11:55:00"

# --- Preprocessing tunables --------------------------------------------------
# A column whose missing-rate differs between pass/fail by at least this much
# (in absolute fraction) earns an explicit "is-missing" indicator feature.
# Grounded in EDA: cols 72/73/345/346 show a ~0.17 gap (51.8% vs 34.6%).
MISSING_INDICATOR_MIN_GAP = 0.05
MAX_MISSING_INDICATORS = 30

# Columns missing in more than this fraction of TRAIN rows are dropped.
MAX_COLUMN_MISSING_FRAC = 0.55

# --- Validation tunables -----------------------------------------------------
N_SPLITS = 5
MIN_TRAIN_FRAC = 0.40   # first fold trains on at least this fraction of history
EMBARGO = 0             # rows to skip between train and test (purge leakage)

# --- Evaluation --------------------------------------------------------------
# Operating point for the detector: "what recall can we get at >= this precision?"
TARGET_PRECISION = 0.25

RANDOM_STATE = 42
