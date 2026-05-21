# Clinical validation (open datasets)

Research validation on open acute-stroke MI data (Liu2024). Not FDA/CE cleared; not a substitute for prospective FMA-linked clinical trials.

## Healthy (healthy)

- Trials: 647, subjects: 40
- MES Cohen's d: 0.8590484863773714
- ensemble_posterior: AUC=0.905, acc=0.646
- MES_0_100: AUC=0.824, acc=0.646
- MES_features_only: AUC=0.969, acc=0.655
- PhysioNet reference cohort (HF cache).
## Stroke (stroke)

- Trials: 826, subjects: 14
- MES Cohen's d: 0.2142054365429433
- Mean Rehab Proxy Index: **34.51**
- ensemble_posterior: AUC=0.528, acc=0.528
- MES_0_100: AUC=0.573, acc=0.557
- MES_features_only: AUC=0.565, acc=0.558
- no_per_subject_clinical_table_rpi_equals_mes_heuristic
- Liu2024 parquet subjects: 14.
