# SECOM Failure Detection + RCA (PoC)

Time-validated failure detection and root-cause analysis on the UCI SECOM
semiconductor sensor dataset.

## Why this is not a vanilla classification problem

| Property | Value | Consequence |
|---|---|---|
| Samples / features | 1567 / 590 | wide, low-n |
| Fail rate | 6.64% (104) | accuracy is useless (all-pass = 93.4%) |
| Missing cells | ~4.5% | imputation is a fit-on-train concern |
| Constant columns | 116 | drop early |
| **Time span** | **86 days, fully ordered** | **random CV leaks the future** |
| **Fail-rate drift** | **14.1% → 3.5% → 5.4%** | **non-stationary; validate walk-forward** |
| Best single-feature AUC | 0.69 (7 feats > 0.65) | many weak signals, not one hero |

## Design decisions (locked)
- **Goal:** detector *and* RCA narrative.
- **Validation:** walk-forward (expanding window). No random k-fold for headline numbers.
- **Metric:** PR-AUC + recall at ≥25% precision. Never accuracy.

## Hypothesis tracker

| ID | Hypothesis | Status |
|----|------------|--------|
| H1 | Walk-forward lowers & stabilizes AUC vs random k-fold | ⬜ |
| H2 | Threshold tuned for recall@precision >> default threshold | ⬜ |
| H3 | Per-column missing-indicators add orthogonal signal | ⬜ (built in `preprocess`) |
| H4 | Correlation-cluster reduction beats all-feature model | ⬜ (`cluster_features`) |
| H5 | Class-weight/threshold beats SMOTE in high-dim sparse space | ⬜ |
| H6 | LightGBM on screened+clustered feats = best PR-AUC | ⬜ |
| H7 | Early-window vs late-window top features differ (drift RCA) | ⬜ |
| H8 | Fail-predictive features concentrate in few tool clusters | ⬜ |
| H9 | PSI/KS drift score is a label-free early-warning monitor | ⬜ |

## Layout
```
src/secom/
  config.py       paths, tunables, operating point
  data.py         load_secom() -> aligned, time-sorted (X, y, timestamps)
  preprocess.py   leak-free Preprocessor (H3) + cluster_features (H4)
  metrics.py      PR-AUC, recall@precision (no accuracy)
  validation.py   walk_forward_splits()  <-- YOUR CONTRIBUTION (see tests)
  features.py     UnivariateScreen + ClusterReducer
  models.py       model factory + fail-scores
  experiment.py   walk-forward runner
  results.py      result persistence and markdown rendering
  rca.py          tool mapping + drift + RCA helpers
  visuals.py      dashboard-ready chart generators
tests/
  test_validation.py   executable spec for the splitter
```

## Environment
```bash
VIRTUAL_ENV=.venv uv venv .venv
VIRTUAL_ENV=.venv .venv/bin/pip install -e .
VIRTUAL_ENV=.venv .venv/bin/pip install lightgbm shap imbalanced-learn matplotlib
```

For reproducible execution from a clean artifact state:
```bash
python -m pytest -q
python experiments/run_all.py
```

## Hypothesis tracker

| ID | Description | Status |
|----|---|---|
| H1 | walk-forward vs random split leakage gap | ✅ |
| H2 | threshold tuning at precision floor | ✅ |
| H3 | missing indicators improve signal | ✅ |
| H4 | cluster reduction | ✅ |
| H5 | class weight vs SMOTE | ✅ |
| H6 | best model family | ✅ |
| H7 | driver drift between early and late windows | ✅ |
| H8 | concentration in few failure clusters | ✅ |
| H9 | PSI as label-free drift monitor | ✅ |
| H10 | expanding vs rolling retraining | ✅ |
| H11 | time-cycle feature lift | ✅ |
| H12 | heterogeneous model ensemble | ✅ |
| H13 | multi-tool workflow simulation | ✅ |
| H14 | late-failure cross-tool RCA | ✅ |
| H15 | dashboard and communication artifacts | ✅ |
| H16 | reproducible report + manifest bundle | ✅ |

## Current delivery files
- `data/raw/manifest.md` (dataset provenance)
- `artifacts/cleaned_secom.parquet` (deterministic cleaned dataset artifact)
- `docs/methodology.md` (assumptions, constraints, rationale)
- `docs/RESULTS.md` (synthesized test outcomes)
- `artifacts/*.png|.json|.csv` (chart outputs and RCA traces)
- `app.py` (Streamlit dashboard launcher)
