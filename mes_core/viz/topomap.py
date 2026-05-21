"""Topomap payloads for the frontend (Plotly renders the actual heatmap)."""

from __future__ import annotations

from typing import Any

import numpy as np

# Standard 10-20 2D positions (normalized to a unit head circle, x right, y up).
# Source: MNE's standard_1020 layout, approximated for the 16-channel OpenBCI set.
DEFAULT_POSITIONS_2D: dict[str, tuple[float, float]] = {
    "Fpz": (0.0, 0.95),
    "Fz":  (0.0, 0.55),
    "FC3": (-0.41, 0.30),
    "FCz": (0.0, 0.30),
    "FC4": (0.41, 0.30),
    "C3":  (-0.55, 0.0),
    "C1":  (-0.28, 0.0),
    "Cz":  (0.0, 0.0),
    "C2":  (0.28, 0.0),
    "C4":  (0.55, 0.0),
    "CP3": (-0.41, -0.30),
    "CPz": (0.0, -0.30),
    "CP4": (0.41, -0.30),
    "T7":  (-0.95, 0.0),
    "T8":  (0.95, 0.0),
    "Pz":  (0.0, -0.55),
}


def scalp_topomap_payload(
    values: np.ndarray,
    ch_names: list[str],
    *,
    positions: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Return a small JSON-serializable payload describing a scalp topomap.

    The frontend renders this with a Plotly scatter + contour over the head.
    """
    positions = positions or DEFAULT_POSITIONS_2D
    out: list[dict[str, Any]] = []
    for name, v in zip(ch_names, values, strict=False):
        if name in positions:
            x, y = positions[name]
            out.append({"channel": name, "x": x, "y": y, "value": float(v)})
    return {
        "points": out,
        "vmin": float(np.nanmin(values)) if len(values) else 0.0,
        "vmax": float(np.nanmax(values)) if len(values) else 1.0,
    }
