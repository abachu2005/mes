# Open datasets for MES

MES is wired for **MOABB** downloads plus your own uploads. This page lists what exists online, what the repo actually uses today, and what is still missing for rehab/FMA-style validation.

## What is on Hugging Face today

[`abachu2005/mes-eeg-processed`](https://huggingface.co/datasets/abachu2005/mes-eeg-processed) currently has **PhysioNet EEG Motor Movement/Imagery only** (~120 parquet files: `physionet_S##_mi.parquet` / `physionet_S##_rest.parquet`).

Stroke parquet (`liu2024_*`, `liu2025_*`) is produced locally via `scripts/preprocess_moabb_datasets.py --stroke` and published with `scripts/publish_hf_stroke_assets.py` (requires `HF_TOKEN`). Per-subject clinical fields: `mes_core/data/liu2024_clinical.tsv` (from Figshare v4 `participants.tsv`, ~3.5 KB).

## Stroke & rehab EEG (open, via MOABB)

| Dataset | MOABB class | N | Paradigm | Clinical notes | MES status |
|---------|-------------|---|----------|----------------|------------|
| **Liu2024** | `Liu2024` | 50 acute stroke | Hand-grip MI (L/R), 29 EEG @ 500 Hz | NIHSS, MBI, mRS in paper + Figshare characteristics; **not FMA** | Download works; preprocess: `python3 scripts/preprocess_moabb_datasets.py --stroke` |
| **Liu2025** | `Liu2025` | 27 stroke | Lower-limb / gait MI, longitudinal sessions | Sci Data 2025; multi-timepoint | Same MOABB path; preprocess `--stroke` includes Liu2025 when implemented |
| Lee2019_MI | `Lee2019_MI` | 54 | Healthy hand MI (OpenBMI) | **Not stroke** — do not use for stroke validation | Was mistakenly referenced in an older preprocess comment |

**Access:** `pip install moabb` then e.g. `from moabb.datasets import Liu2024; Liu2024().get_data(subjects=[1])` (first run downloads ~463 MB from Figshare).

**Paper:** Liu et al., *Scientific Data* 2024 — [10.1038/s41597-023-02787-8](https://doi.org/10.1038/s41597-023-02787-8)

## Healthy / BCI benchmarks (open, via MOABB)

| Dataset | Use in MES |
|---------|------------|
| PhysioNet EEGMMIDB | Production training & `mes validate` (on HF) |
| BCI Competition IV 2a/2b | `scripts/preprocess_moabb_datasets.py --bci-iv-2b` |
| Lee2019 / BNCI suites | Optional extra benchmarks |

## Rehab outcomes (FMA, ARAT, etc.)

Open EEG sets rarely ship **Fugl-Meyer (FMA)** in the same table as trial-level EEG. Practical options:

1. **Liu2024 patient characteristics** on Figshare (NIHSS, MBI, mRS, hemiplegia side, days post-stroke) — correlate with per-subject mean MES via `mes_core/eval/outcomes.py` after you build a CSV (`participant_code,fma` or `nihss`, etc.).
2. **Your IRB pilot** — upload OpenBCI `.txt` + outcomes CSV; recovery index in the app uses prior sessions, not FMA automatically.
3. **Newer multimodal sets** (e.g. post-stroke fNIRS+EEG with FMA/ARAT on Figshare, 2025–2026) — not integrated in this repo yet; good candidate for a follow-on ingest script.

There is **no** large public “stroke rehab EEG + FMA longitudinal” set wired into MES like PhysioNet is today.

## Commands to add stroke data to the pipeline

```bash
pip install -e ".[train,dev]"   # includes moabb
python3 scripts/preprocess_moabb_datasets.py --stroke --out /tmp/mes-out/processed
python3 scripts/fit_mes_weights.py --data-dir /tmp/mes-out/processed --cohort stroke
mes validate --data-dir /tmp/mes-out/processed --max-files 60
# Upload to HF when ready:
# python3 scripts/hf_sync.py --repo-id abachu2005/mes-eeg-processed --subdir processed ...
```

## Honest gap

- **Product:** cohort toggle `healthy` | `stroke` and recovery z-score vs past sessions.
- **Science:** stroke validation on HF is still **planned**, not shipped; rehab proxy is research-only until Liu parquet + optional outcomes CSV (or prospective data) are in the loop.
