"""Shared helpers for locating chained Kaggle kernel inputs (no HF calls)."""

from __future__ import annotations

from pathlib import Path


def find_parquet_dir() -> Path:
    """Locate processed parquet files from a chained preprocess kernel."""
    candidates: list[Path] = []

    inp = Path("/kaggle/input")
    if inp.exists():
        for sub in sorted(inp.iterdir()):
            proc = sub / "processed"
            if proc.is_dir() and list(proc.glob("*.parquet")):
                candidates.append(proc)
            if list(sub.glob("physionet_*.parquet")):
                candidates.append(sub)
            if list(sub.rglob("physionet_*.parquet")):
                candidates.append(sub)

    local = Path("/kaggle/working/processed")
    if local.is_dir() and list(local.glob("*.parquet")):
        candidates.append(local)

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No processed parquet found. Attach kernel abachu2005/mes-00-preprocess as a data source."
    )


def find_onnx_files() -> list[Path]:
    """Locate ONNX model files from chained training kernels."""
    found: list[Path] = []
    inp = Path("/kaggle/input")
    if inp.exists():
        found.extend(sorted(inp.rglob("*.onnx")))
    found.extend(sorted(Path("/kaggle/working").glob("*.onnx")))
    # Deduplicate by name
    seen: set[str] = set()
    out: list[Path] = []
    for p in found:
        if p.name not in seen:
            seen.add(p.name)
            out.append(p)
    return out
