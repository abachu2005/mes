# Validation

All targets below are **honestly reported, not hard pass/fail gates**. The
auto-generated [`benchmarks.md`](../benchmarks.md) on the model repo is the
source of truth.

## Cross-validation

- **Within-subject 5-fold CV** on PhysioNet eegmmidb. Reported per-subject
  accuracy and AUC.
- **Leave-one-subject-out (LOSO)** across PhysioNet + BCI IV 2a. Reported as
  the production model's headline metric.

## Reliability

- **ICC(3,1)** of MES scores across two recording sessions per subject (where
  multiple sessions are available in the open datasets). Target: ICC > 0.7
  on PhysioNet.

## Stroke validation

- **Liu2024** (50 acute stroke patients, hand MI): per-subject MES + accuracy.
- **Liu2025** (27 stroke patients, longitudinal gait MI): exploratory analysis
  of MES trend across recording sessions. Not powered for statistical claims.

## Calibration

- **Brier score** and reliability diagram on held-out PhysioNet test
  participants. Production model uses Platt scaling.

## Reproducibility

Every benchmark is computed by `notebooks/kaggle/03_validate.py` against a
pinned commit SHA on the HF dataset + model repos.
