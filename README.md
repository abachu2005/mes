# MES — Motor Engagement Signal

> **Quantifying neural drive for movement recovery.**

MES is a research-grade, end-to-end pipeline that takes recorded EEG from a participant performing motor imagery or movement tasks and produces a single, calibrated score (the Motor Engagement Signal, 0–100) reflecting the strength of motor-cortical engagement. It is designed for tracking rehabilitation progress in stroke and spinal-cord-injury patients.

> **Research use only — not FDA / CE cleared. Do not enter PHI.**

## What's in this repo

```
mes_core/        installable Python package (preprocessing, features, scoring, eval, viz)
backend/         FastAPI app (single-container, SQLite, HF Hub storage)
frontend/        React + Vite + Tailwind dashboard (demo-ready)
notebooks/kaggle/ training + validation notebooks (run headlessly on Kaggle)
tests/           pytest + hypothesis + Playwright (synthetic + 2-subject BCI IV 2b fixtures)
docs/            MkDocs site (methods, benchmarks, hardware SOP)
```

## Live demo

[https://huggingface.co/spaces/abachu2005/mes](https://huggingface.co/spaces/abachu2005/mes)

> First request after 48 h of inactivity may take 1–2 min to warm up (HF Space free-tier cold start).

## How it works

1. **Upload** an EEG recording (EDF / BDF / OpenBCI `.txt`).
2. **Preprocess**: bandpass 0.5–40 Hz, ICA artifact removal, spatial mapping to the 16-channel OpenBCI Cyton+Daisy montage at 125 Hz, cue-locked epoching.
3. **Extract features**: mu/beta band power, ERD%, MRCP amplitude, lateralization index, Riemannian covariances.
4. **Classify** the trial vs rest with two models (Riemannian + Logistic Regression, and EEGNet v4).
5. **Combine** features + classifier confidence into the MES score:

   ```
   raw = w1·z(ERD_mu) + w2·z(ERD_beta) + w3·z(LI) + w4·z(MRCP) + w5·logit(p_model)
   MES = 100 · sigmoid(raw)
   ```

6. **Report**: time-resolved MES trace, ERD topomap, lateralization summary, downloadable PDF, longitudinal chart across sessions.

## Data

All open-source datasets, loaded via [MOABB](https://moabb.neurotechx.com):

- PhysioNet EEG Motor Movement/Imagery (109 subjects)
- BCI Competition IV 2a + 2b
- Liu2024 — 50 acute stroke patients (hand MI)
- Liu2025 — 27 stroke patients (gait MI, longitudinal)

Processed datasets are published to [`abachu2005/mes-eeg-processed`](https://huggingface.co/datasets/abachu2005/mes-eeg-processed). Trained ONNX models live at [`abachu2005/mes-models`](https://huggingface.co/abachu2005/mes-models). See [`docs/benchmarks.md`](docs/benchmarks.md) for the validation results.

## Hardware target

OpenBCI Cyton + Daisy (16 channels @ 125 Hz). Electrode placement and recording SOP: [`docs/hardware.md`](docs/hardware.md).

## Quickstart (developer)

```bash
# Python 3.11 required.
make install-dev
make test
mes score path/to/recording.edf    # CLI — ONNX ensemble + fitted weights
mes validate --download            # benchmark MES vs posterior-only baselines
```

See [`docs/cli.md`](docs/cli.md) and [`docs/joss-roadmap.md`](docs/joss-roadmap.md) for JOSS submission planning.

Run the app locally (Docker):

```bash
make docker-run     # builds image and serves on http://localhost:7860
```

## Limitations

- 16-channel mapped data does not match 64+ channel research caps for cross-subject MI; benchmark gap is reported in `docs/benchmarks.md`.
- Stroke validation is limited to two open datasets (Liu2024, Liu2025); clinical deployment would require IRB-approved prospective data.
- HF Space free-tier sleeps after 48 h of inactivity.
- ICA quality degrades at low channel counts; we run ICA on the full source montage before downsampling to mitigate this.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
