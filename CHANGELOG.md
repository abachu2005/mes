# Changelog

All notable changes to the `mes` Python package are documented here.

## [Unreleased]

## [0.2.0] - 2026-08-20

First citable release. Archived on Zenodo.

### Added

- Bundled fitted MES weights (`mes_core/data/mes_weights_right_hand.json`).
- `mes_core.pipeline` with `score_recording` / `score_epochs` (ONNX ensemble + fitted weights).
- CLI: `mes score`, `mes validate`, `mes fit-weights`.
- `mes_core.eval.validate` harness and `scripts/fit_mes_weights.py`.
- Rest-block heuristics (`mes_core.scoring.rest`) and a 60 s protocol rest block
  for sliding-window uploads.
- Sliding-window epoching for continuous recordings without annotations.
- Rehab cohort support: recovery proxy index (RPI), stroke-fitted weights, and
  Liu2024 validation.
- Signal quality gates and the protocol UI.
- Guide to open stroke/rehab EEG datasets.
- End-to-end autonomous protocol test script and API/CLI coverage.
- JOSS submission materials: `paper.md`, `paper.bib`, `docs/joss-roadmap.md`.
- Citation metadata: `CITATION.cff` and `.zenodo.json`.

### Changed

- Backend session pipeline uses `score_epochs` (parity with CLI).
- CI lint scope passes with updated ruff configuration.
- Authorship and affiliations finalized across `CITATION.cff`, `paper.md`,
  `pyproject.toml`, and `LICENSE`.

## [0.1.0] - 2026-05-01

- Initial public release: preprocessing, MES scoring, FastAPI backend, React UI, HF training pipeline.
