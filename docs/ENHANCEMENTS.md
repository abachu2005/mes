# MES enhancement stack (v0.2+)

## Production scoring

- **Quality gates** (`mes_core.quality`) — reject or flag flatline / noisy epochs
- **Reliability tier** — High / Medium / Low on every session report
- **Cohort weights** — `healthy` vs `stroke` bundles in `mes_core/data/`
- **Recovery index** — `mes_recovery_z` vs participant's prior sessions
- **Protocol UI** — checklist + rest-block confirmation on upload

## Validation & training

```bash
mes validate --download --max-files 60
python scripts/eval_model_quality.py --download
python scripts/fit_mes_weights.py --download --cohort stroke
python scripts/preprocess_moabb_datasets.py --out /tmp/mes-out/processed
```

## Clinical outcomes (CSV)

Provide `participant_code,fma,arat` CSV and use `mes_core.eval.outcomes` to correlate with mean MES.

## Next science steps

1. Upload Liu/Lee2019 parquet to HF `processed/`
2. Refit stroke weights + re-run validate
3. Prospective IRB pilot linking ΔMES to ΔFMA
