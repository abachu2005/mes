# AGENTS.md

## Cursor Cloud specific instructions

- **Python:** use `python3` (no `python` on PATH).
- **Install deps:** `pip install -e ".[dev]"` from repo root (see `pyproject.toml`).
- **Unit tests:** `python3 -m pytest tests/ -q --ignore=tests/integration`
- **Autonomous protocol E2E:** `python3 scripts/run_protocol_e2e.py --live https://abachu2005-mes.hf.space --json-out /tmp/report.json` (local scoring + optional live API upload).
- **PhysioNet sanity:** `mes validate --download --max-files 30` (expect ensemble AUC ~0.93).
- **Live demo:** [HF Space v0.2](https://huggingface.co/spaces/abachu2005/mes) — cold start after ~48 h idle can take 1–2 min.
- **Synthetic OpenBCI uploads** get correct rest/task split and **High** reliability with `had_rest_block=true`, but absolute MES often stays low (~2–20) because ONNX was trained on PhysioNet parquet, not synthetic sine/pink noise. For “good” MES (~40–80), use labeled parquet / real MI recordings per `docs/hardware.md`.
