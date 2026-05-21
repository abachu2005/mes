# AGENTS.md

## Cursor Cloud specific instructions

### Overview

MES (Motor Engagement Signal) is a research-grade EEG analysis pipeline with:
- **Backend**: FastAPI (Python 3.11) on port 7860, with embedded SQLite
- **Frontend**: React + Vite + Tailwind, dev server on port 5173 (proxies `/api` to 7860)

### Running services

| Service | Command | Port |
|---------|---------|------|
| Backend | `python3.11 -m backend.app.main` | 7860 |
| Frontend dev | `cd frontend && npm run dev` | 5173 |

In production mode, `npm run build` in `frontend/` outputs to `backend/app/static/` and the backend serves both API and SPA.

### Key commands

See the `Makefile` for the full list. The most useful ones:

- **Install dev deps**: `make install-dev`
- **Tests**: `make test` (pytest with coverage)
- **Fast tests**: `make test-fast` (unit + property tests only)
- **Lint**: `make lint` (ruff + mypy); note: there are pre-existing lint issues
- **Format**: `make format` (auto-fix with ruff)
- **Frontend lint**: `cd frontend && npm run lint`
- **Frontend build**: `cd frontend && npm run build` (or `make frontend-build`)
- **Smoke test**: `make smoke` (boots backend, verifies app object builds)

### Gotchas

- **Python 3.11 required**: The `Makefile` hard-codes `PYTHON ?= python3.11`. Python 3.12+ is not sufficient because the system numpy from 3.12 conflicts with 3.11's C extensions. After installing Python 3.11 from deadsnakes PPA, you must force-install numpy to the 3.11 user site: `python3.11 -m pip install "numpy>=1.26,<2.2"`.
- **edfio**: The integration test `test_session_upload_and_pipeline_round_trip` requires the `edfio` package (MNE optional dep for EDF export). Install with `python3.11 -m pip install edfio`.
- **PATH**: pip installs scripts to `/home/ubuntu/.local/bin` which may not be on PATH. Export it: `export PATH="/home/ubuntu/.local/bin:$PATH"`.
- **System libs for WeasyPrint**: `libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf2.0-0 libffi-dev fonts-dejavu` must be installed for PDF report generation.
- **Demo data**: Seed demo sessions via `curl -X POST http://localhost:7860/api/demo/seed` for a quick populated dashboard.
- **Health endpoint**: The API health check is at `/api/healthz` (not `/api/health`).
- **SPA catch-all**: The backend's catch-all route serves `index.html` for any non-file path. API routes are prefixed with `/api/`.
- **No HF_TOKEN needed for dev**: Without a Hugging Face token, the backend falls back to heuristic scoring (no ONNX model downloads). All tests pass without it.
