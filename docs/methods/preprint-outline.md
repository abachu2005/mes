# Methods preprint outline (companion to JOSS)

**Working title:** *Motor Engagement Signal (MES): an interpretable EEG composite for motor-imagery rehabilitation research*

## Abstract (draft bullets)

- Problem: BCI studies report accuracy; rehab trials need comparable engagement endpoints.
- Solution: MES fuses ERD, lateralization, MRCP, and ensemble classifier posteriors into [0, 100].
- Data: PhysioNet MI, BCI IV, Liu2024 acute stroke MI, Liu2025 longitudinal gait MI.
- Results: MES separates task vs rest with effect sizes; comparison to posterior-only decoding.

## Sections

1. Introduction — stroke MI, low-channel ambulatory EEG.
2. Related work — MOABB benchmarks, SMR-BCI rehab trials, Liu datasets.
3. MES definition — equation, anti-circular weight fitting, subject vs population baseline.
4. Preprocessing — ICA policy, montage mapping, epoch window (750 samples).
5. Classifiers — Riemannian + EEGNet ensemble, ONNX export.
6. Validation — `mes validate` metrics, stroke held-out splits (when parquet available).
7. Limitations — 16ch vs 64ch cap, research-only, ICA at low density.
8. Software availability — GitHub, HF Hub, editable install / `motor-engagement-signal`, JOSS citation.

## Figures to generate

- Pipeline diagram (upload → MES → report).
- MES distribution task vs rest (PhysioNet + Liu).
- Longitudinal example (Liu2025 or synthetic demo).
- Comparison bar chart: `ensemble_posterior` vs `MES_0_100` AUC.

## Target venues

- arXiv (q-bio.NC or eess.SP) before JOSS acceptance.
- Optional: *Journal of NeuroEngineering and Rehabilitation* methods brief.
