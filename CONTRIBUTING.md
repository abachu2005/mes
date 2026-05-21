# Contributing to MES

Thank you for helping improve the Motor Engagement Signal (MES) project.

## Development setup

```bash
make install-dev
make test
```

Python **3.11** is required. See `AGENTS.md` (if present) for cloud VM notes.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Add tests for new behavior in `tests/`.
3. Run `ruff check mes_core backend tests` and `pytest`.
4. Update `CHANGELOG.md` under **Unreleased** for user-visible changes.
5. Open a PR with a clear description and link any related issue.

## Reporting issues

Use GitHub Issues for bugs, documentation gaps, and feature requests. Include:

- MES version (`mes version`)
- Python version
- Minimal steps to reproduce (file format, command, error trace)

## Scope

MES is **research software only** (not a medical device). Contributions that improve
reproducibility, validation, documentation, and open science are especially welcome.

## JOSS / citation

If you use MES in publication, cite the software archive DOI (after Zenodo release)
and the companion methods preprint when available. See `CITATION.cff` and `paper.md`.
