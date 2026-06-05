# SECOM Failure Detection + RCA (PoC)

Time-validated anomaly/failure detection and cross-tool root-cause tracing on the UCI SECOM semiconductor sensor dataset.

## Why this is not a vanilla classification problem

| Property | Value | Consequence |
|---|---|---|
| Samples / features | 1567 / 590 | wide, low-n |
| Fail rate | 6.64% (104) | accuracy is misleading (all-pass = 93.4%) |
| Missing cells | ~4.5% | imputation must be fit within training folds |
| Constant columns | 116 | drop early |
| **Time span** | **86 days, fully ordered** | **random CV leaks the future** |
| **Fail-rate drift** | **14.1% → 3.5% → 5.4%** | **non-stationary; walk-forward is required** |
| Best single-feature AUC | 0.69 (7 feats > 0.65) | no single dominant feature | 

## Design decisions (locked)

- **Goal:** build a production-safe failure detector plus cross-tool RCA narrative.
- **Validation:** walk-forward expanding splits for all model performance reporting.
- **Primary metrics:** PR-AUC and recall at a target precision floor.
- **Workflow:** simulated multi-tool chain `Deposition -> Etch -> Inspection`.

## Hypothesis status and latest outcomes (H1–H16)

Implemented in this repo:

| Hypothesis | Best configuration | Key result |
|---|---|---|
| H1 | walk-forward vs random_kfold | 0.052 vs 0.118 PR-AUC (random is over-optimistic) |
| H2 | threshold tuning | tuned @25% precision improves recall (`0.000 -> 0.026`) |
| H3 | missing indicators | indicators reduced headline PR-AUC (`0.089 -> 0.086`) |
| H4 | feature clustering | all-features better than clustered (`0.086 vs 0.069`) |
| H5 | class-weight/SMOTE | SMOTE improved PR-AUC (`0.101`) over class-weight (`0.086`) |
| H6 | best model family | LightGBM (`lgbm_all`, PR-AUC `0.086`, recall@25 `0.193`) |
| H7 | driver stability | low early/late overlap indicates feature drift |
| H8 | failure-cluster concentration | signal concentrates in a smaller cluster set |
| H9 | PSI drift monitor | weak/noisy label-free trend (`rho -0.378`, `p 0.2518`) |
| H10 | retraining strategy | expanding window better (`0.086` vs rolling30 `0.067`) |
| H11 | time-cycle features | cycle features improved (`0.095` vs `0.086`) |
| H12 | heterogeneous blend | blend underperformed single LightGBM (`0.057` vs `0.103`) |
| H13 | stage simulation | inspection stage best among cumulative stages (`0.089`) |
| H14 | late defect attribution | 15 late cases traced; top vote = deposition |
| H15 | dashboard outputs | chart and JSON deliverables generated |
| H16 | reproducibility | manifest + run metadata bundle generated |

## Cross-tool RCA approach (brief)

1. Partition features into simulated tool stages in process order:
   - **Deposition:** `f0-199`
   - **Etch:** `f200-399`
   - **Inspection:** `f400-589`
2. Train stage-wise models with walk-forward splits (`Deposition -> Etch -> Inspection`) and score late-stage risk with cumulative feature context.
3. For each high-risk late defect, compute per-feature contribution evidence.
4. Aggregate contributions by stage/tool map to produce a ranked candidate root-cause path.
5. Emit repeatable RCA payloads with top features, top tool votes, and case narratives.

This gives a clear traceability chain: late defect alert → likely upstream contributors → review candidates for Deposition/Etch/Inspection.

## Project structure

```text
src/secom/
  config.py       paths, tunables, operating point
  data.py         load_secom() -> aligned, time-sorted (X, y, timestamps)
  preprocess.py   leak-safe preprocessing + optional missing indicators + clustering hook
  metrics.py      PR-AUC, recall@precision
  validation.py   walk_forward_splits() (primary split contract)
  features.py     UnivariateScreen + ClusterReducer
  models.py       model factory and scoring
  experiment.py   shared experiment runner
  results.py      result persistence + markdown render
  rca.py          tool mapping + attribution helpers
  visuals.py      chart generation for dashboard/reporting

tests/
  test_validation.py   split contract and leakage checks
  test_experiment.py   metrics and runner contract
  test_models.py       baseline model checks
  test_rca.py          attribution pipeline checks
```

## Setup and execution

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r pyproject.toml
```

Run reproducibility pipeline:

```bash
python -m pytest -q
python experiments/run_all.py
python -m streamlit run app.py
```

## Deliverables

- `data/raw/manifest.md` — source provenance metadata
- `artifacts/cleaned_secom.parquet` — deterministic cleaned dataset export
- `docs/methodology.md` — assumptions + preprocessing + evaluation logic
- `docs/RESULTS.md` — full H1–H16 experiment table
- `artifacts/` PNG/JSON deliverables (visuals + RCA case outputs)
- `docs/SECOM_Complete_Report.docx` — consolidated results report for external sharing
