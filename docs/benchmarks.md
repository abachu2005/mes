# Benchmarks

This page is **auto-generated** by `notebooks/kaggle/03_validate.py`.

When training has completed on Kaggle, the latest results are published to
[`benchmarks.md`](https://huggingface.co/abachu2005/mes-models/blob/main/benchmarks.md)
on the HF model repo. The frontend's About page also pulls the latest
benchmarks JSON live.

## Live results

Until the first training run completes, this page is intentionally a stub.
Once `03_validate.py` finishes:

| Dataset | Model | n | Accuracy | AUC | Brier |
|---|---|---:|---:|---:|---:|
| _(populated by the validation notebook)_ | | | | | |

## How to refresh

```bash
python scripts/kaggle_submit.py notebooks/kaggle/03_validate.py \
    --kernel-id abachu2005/mes-03-validate \
    --no-gpu --internet --poll
```

This pulls the latest `mes-eeg-processed` dataset SHA and `mes-models`
model SHA, evaluates every ONNX file present, and pushes back
`benchmarks.md` + `benchmarks.json`.
