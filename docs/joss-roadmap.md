# JOSS submission roadmap

This document tracks readiness for the [Journal of Open Source Software](https://joss.theoj.org/).

## Desk-rejection gates

| Requirement | Status | Notes |
|-------------|--------|-------|
| OSI license (`LICENSE`) | Done | Apache-2.0 |
| 6+ months public development | **Action** | Maintain open commits/issues until eligible |
| Research impact evidence | **Action** | Methods preprint + external pilot users |
| Tests + CI | Done | `pytest`, GitHub Actions |
| CONTRIBUTING + CoC | Done | Root-level files |
| Tagged releases | **Action** | Create `v0.2.0` on GitHub after merge |

## Software scope (center the library)

JOSS reviews **`mes_core`**, not the HF Space alone:

```bash
pip install -e .
mes score recording.edf
mes validate --download
```

Pre-trained ONNX on the Hub are **artifacts**, not the submission itself.

## Paper checklist (`paper.md`)

- [x] Summary
- [x] Statement of need
- [x] State of the field
- [x] Software design
- [x] Research impact (update with preprint DOI before submit)
- [x] AI usage disclosure
- [ ] Author affiliations finalized
- [ ] Zenodo archive DOI after acceptance

## Scholarly novelty (companion methods paper)

Document separately from JOSS:

1. MES composite index (anti-circular weight fitting).
2. OpenBCI 16-channel contract + ensemble ONNX inference.
3. Stroke dataset validation (Liu2024/2025) vs accuracy-only BCI metrics.

See `docs/methods/preprint-outline.md`.

## Commands for reviewers

```bash
make install-dev
pytest
mes version
mes score tests/fixtures/sample.edf   # when fixture exists
mes validate --download --max-files 15
```

## pyOpenSci (optional)

Consider [pyOpenSci](https://pyopensci.org/) packaging review before JOSS submission.
