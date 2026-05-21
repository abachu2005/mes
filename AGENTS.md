# AGENTS.md

## Cursor Cloud specific instructions

- **Python:** use `python3` (no `python` on PATH).
- **Install deps:** `pip install -e ".[dev]"` from repo root (see `pyproject.toml`).
- **Unit tests:** `python3 -m pytest tests/ -q --ignore=tests/integration`
- **Protocol upload test:** `python3 scripts/generate_protocol_test_file.py -o /tmp/protocol.txt` then `python3 -c "from mes_core.pipeline import score_recording; print(score_recording('/tmp/protocol.txt', had_rest_block=True).to_dict())"`
- **Live demo:** [HF Space v0.2](https://huggingface.co/spaces/abachu2005/mes) — cold start after ~48 h idle can take 1–2 min.
- **Synthetic OpenBCI uploads** get correct rest/task split and **High** reliability with `had_rest_block=true`, but absolute MES often stays low (~2–20) because ONNX was trained on PhysioNet parquet, not synthetic sine/pink noise. For “good” MES (~40–80), use labeled parquet / real MI recordings per `docs/hardware.md`.
