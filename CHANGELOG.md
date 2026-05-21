# Changelog

All notable changes to the `mes` Python package are documented here.

## [Unreleased]

## [0.2.1] - 2026-05-21

### Added

- MIT license, [`.zenodo.json`](.zenodo.json), and [`docs/releasing.md`](docs/releasing.md) for GitHub → Zenodo archiving.
- Citation block in README; distribution table (GitHub / HF / Docker, no PyPI).

### Changed

- License changed from Apache-2.0 to MIT across `LICENSE`, `pyproject.toml`, and `CITATION.cff`.

## [0.2.0] - 2026-05-21

### Added

- Stroke pipeline: Liu2024 preprocess, stroke ONNX, benchmarks, clinical TSV, CI gates.
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
