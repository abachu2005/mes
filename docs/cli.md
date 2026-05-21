# Command-line interface

Install the package, then use the `mes` entrypoint:

```bash
pip install -e .
mes version
```

## Score a recording

Uses production ONNX ensemble (when available) and fitted weights from
`mes_core/data/mes_weights_right_hand.json`:

```bash
mes score path/to/recording.edf --task right_hand --out result.json
```

Options:

- `--no-onnx` — heuristic posterior only (offline / no Hub access).

## Validate on processed data

Compare MES to classifier-only baselines on labeled parquet:

```bash
mes validate --download --max-files 30 --out-dir validation_out
```

Outputs `validation_report.json` and `validation_report.md`.

## Fit weights

Re-fit the bundled weights from Hugging Face processed data:

```bash
python scripts/fit_mes_weights.py --download --max-files 40
# optional: --upload with HF_TOKEN set
```

Or via CLI:

```bash
mes fit-weights --download --max-files 40
```
