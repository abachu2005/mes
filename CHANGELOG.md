# Changelog

All notable changes to the `mes` Python package are documented here.

## [Unreleased]

### Added

- Bundled fitted MES weights (`mes_core/data/mes_weights_right_hand.json`).
- `mes_core.pipeline` with `score_recording` / `score_epochs` (ONNX ensemble + fitted weights).
- CLI: `mes score`, `mes validate`, `mes fit-weights`.
- `mes_core.eval.validate` harness and `scripts/fit_mes_weights.py`.
- Rest-block heuristics (`mes_core.scoring.rest`).
- JOSS submission materials: `paper.md`, `paper.bib`, `CITATION.cff`, `docs/joss-roadmap.md`.

### Changed

- Backend session pipeline uses `score_epochs` (parity with CLI).
- CI lint scope passes with updated ruff configuration.

## [0.1.0] - 2026-05-01

- Initial public release: preprocessing, MES scoring, FastAPI backend, React UI, HF training pipeline.
