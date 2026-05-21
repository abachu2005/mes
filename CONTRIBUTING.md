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

## Releases and citation

We release via **GitHub tags**; **Zenodo** mints a DOI per release (see `docs/releasing.md`).
PyPI and Bioconda are intentionally not used.

If you use MES in a publication, cite the Zenodo DOI for the version you used
(update `CITATION.cff` after the first archived release). See also `paper.md` for JOSS planning.
