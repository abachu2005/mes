# Benchmarks

This page is **auto-generated** by `notebooks/kaggle/03_validate.py`.

When training has completed on Kaggle, the latest results are published to
[`benchmarks.md`](https://huggingface.co/abachu2005/mes-models/blob/main/benchmarks.md)
on the HF model repo. The frontend's About page also pulls the latest
benchmarks JSON live.

## Live results (training LOSO)

Published automatically to the [model repo](https://huggingface.co/abachu2005/mes-models)
via `scripts/publish_benchmarks.py` after EEGNet training completes:

| Model | Task | n_train | LOSO accuracy | Trained on |
|---|---|---:|---:|---|
| `riemannian_lr_right_hand.onnx` | right_hand_vs_rest | 967 | 0.444 | github_actions |
| `eegnet_right_hand.onnx` | right_hand_vs_rest | 967 | 0.693 | hf_jobs |

Held-out per-dataset metrics (PhysioNet / BCI / Liu) from `03_validate.py`:

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
