# Clinical validation (open data)

MES uses **open acute-stroke MI data (Liu2024)** plus **healthy PhysioNet MI** for research validation. This is **not** FDA/CE clearance and **not** a substitute for a prospective FMA-linked trial.

## Rehab Proxy Index (RPI)

For `cohort=stroke` uploads and Liu2024 benchmarks:

1. **Paretic-hand MES** — mean MES on motor-imagery trials for the paralyzed side (when `liu2024_clinical.tsv` lists `paralysis_side`).
2. **Capacity weight** — `(MBI/100) × (1 − NIHSS/42)`, clipped, when clinical fields exist.
3. **RPI** = paretic-hand MES × capacity weight.

Without clinical TSV, RPI equals pooled MI MES (all labeled hand trials).

## Run locally

```bash
# 1) Preprocess Liu2024 (MOABB; ~463 MB EDF zip + time per subject)
python3 scripts/preprocess_moabb_datasets.py --stroke --out data/processed_stroke

# 2) Optional: patient NIHSS/MBI table (~1.8 GB sourcedata.zip once)
python3 scripts/fetch_liu2024_clinical.py --download

# 3) Fit stroke weights + validation report
python3 scripts/run_clinical_validation.py --data-dir data/processed_stroke --out validation_out/clinical
```

Reports: `validation_out/clinical/clinical_validation.json` and `.md`.

## Latest snapshot (partial Liu2024, 14/50 subjects)

| Cohort | MES AUC (task vs rest) | Cohen's d | Notes |
|--------|------------------------|-----------|--------|
| Healthy PhysioNet | ~0.82–0.97 | ~0.86 | Production reference |
| Stroke Liu2024 | ~0.55–0.58 | ~0.21 | Healthy-trained ONNX transfers poorly; stroke weights emphasize ERD/LI |

**Next engineering steps for stronger stroke performance:**

- Finish all 50 Liu2024 subjects in `mes-eeg-processed` on HF.
- Fine-tune ONNX on stroke parquet (or train stroke-only Riemannian head).
- Upload `liu2024_clinical.tsv` and report Spearman(MES, MBI/NIHSS).
- Prospective IRB cohort with FMA/ARAT and OpenBCI protocol uploads.

## Honest label

**Clinically validated** in the regulatory sense requires prospective outcome evidence. **Open-dataset clinically oriented validation** is what this repo implements today.
