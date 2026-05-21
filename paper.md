---
title: 'MES: Motor Engagement Signal software for open EEG rehabilitation research'
tags:
  - Python
  - EEG
  - neurorehabilitation
  - motor imagery
  - BCI
authors:
  - name: MES Project Authors
    affiliation: 1
affiliations:
  - name: To be completed by submitting authors
    index: 1
date: 21 May 2026
bibliography: paper.bib
---

# Summary

The Motor Engagement Signal (MES) is an open-source Python library and companion
web application that converts clinical-style EEG recordings into a single
calibrated score (0–100) reflecting motor-cortical engagement during movement or
motor-imagery tasks. MES targets rehabilitation researchers who need interpretable,
session-comparable endpoints rather than raw decoding accuracy alone. The software
integrates preprocessing for low-channel ambulatory hardware (OpenBCI Cyton+Daisy),
interpretable neurophysiological features (mu/beta event-related desynchronization,
lateralization, movement-related cortical potentials), and an ensemble of Riemannian
and deep-learning classifiers exported for reproducible ONNX inference.

# Statement of need

Stroke and spinal-cord injury rehabilitation studies increasingly use EEG-based
brain–computer interfaces and motor-imagery training, but published toolchains
typically report classification accuracy or band-power maps in isolation. Clinicians
and trialists tracking recovery over weeks need a **bounded, comparable scalar**
that combines multiple neurophysiological signatures and model confidence without
circularly training on classifier outputs. Existing general-purpose packages
(MNE-Python, MOABB, PyRiemann, Braindecode) provide excellent building blocks but
do not ship an integrated Motor Engagement index, OpenBCI-first montage contract,
longitudinal reporting, and reproducible model artifacts for ambulatory 16-channel
systems. MES fills that gap as installable research software (`pip install mes`)
with a documented CLI, validation harness, and optional FastAPI dashboard.

# State of the field

| Software | Primary contribution | Gap relative to MES |
|----------|---------------------|---------------------|
| MNE-Python [@mne] | General M/EEG analysis | No rehab-specific 0–100 engagement index |
| MOABB [@moabb] | Benchmark ML pipelines | No interpretable composite score + reporting |
| PyRiemann [@pyriemann] | Riemannian geometry classifiers | No fusion with ERD/LI/MRCP + EEGNet ensemble |
| Braindecode [@braindecode] | Deep learning on EEG | No stroke-longitudinal scoring layer |
| py_neuromodulation [@pyneuro] | Feature + decode toolbox | Not OpenBCI-montage + MES pipeline |

MES **extends** these ecosystems rather than reimplementing them: preprocessing
uses MNE; Riemannian features use PyRiemann; EEGNet training uses Braindecode;
MES adds the scoring layer, weight fitting against dataset labels, ONNX deployment,
and validation tooling.

# Software design

`mes_core` is layered for reuse outside the web UI:

1. **I/O** — EDF/BDF/OpenBCI loaders and Hugging Face Hub artifacts.
2. **Preprocessing** — bandpass/notch, ICA when channel count allows, spherical
   mapping to a fixed 16-channel montage at 125 Hz, cue-locked epochs.
3. **Features** — ERD, lateralization, MRCP amplitudes.
4. **Models** — ONNX ensemble (Riemannian tangent-space logistic regression +
   EEGNet v4) with automatic 750-sample window alignment.
5. **Scoring** — `compute_mes` combines z-scored features and `logit(p_model)`
   with weights fit via logistic regression on **dataset-provided** task/rest
   labels (anti-circular design).
6. **Pipeline** — `score_recording` / `score_epochs` for CLI and backend parity.
7. **Evaluation** — `mes validate` compares MES to posterior-only baselines on
   processed parquet.

The FastAPI/React Space is a thin deployment surface; reviewers can verify core
functionality with `mes score recording.edf` without Docker.

# Research impact statement

- **Reproducible artifacts** on Hugging Face Hub: processed parquet, ONNX models,
  and benchmarks JSON published from CI.
- **Fitted calibration bundle** (`mes_weights_right_hand.json`) derived from
  PhysioNet motor-imagery parquet with documented LOSO training metrics for
  constituent classifiers.
- **Live demo** Space for hands-on evaluation by non-specialists.
- **Companion methods manuscript** (in preparation) will report stroke-dataset
  validation (Liu2024/2025) and comparison to accuracy-only endpoints; this JOSS
  submission describes the software enabling that work.

External adoption is growing through open repositories and coursework demos;
authors will update this section with preprint DOI and collaborator citations
prior to final JOSS acceptance.

# AI usage disclosure

Generative AI tools assisted with boilerplate documentation, CI workflow edits,
and portions of the React dashboard. All scientific design choices (MES formula,
anti-circular weight fitting, ensemble inference, validation metrics) were made
and reviewed by human authors. AI-generated code was run through the project test
suite (`pytest`) and manual review before merge.

# References
