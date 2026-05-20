"""Centralized configuration constants for MES.

All shared constants live here so the inference path and training notebooks
agree on the exact preprocessing + model contract.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 16-channel OpenBCI Cyton+Daisy sensorimotor-centered montage (10-20 system).
# Order matters - it defines the channel axis for ONNX models.
OPENBCI_MONTAGE_16: tuple[str, ...] = (
    "Fpz",  # frontal reference (eye artifact regression)
    "Fz",   # frontal reference
    "FC3", "FCz", "FC4",  # premotor
    "C3", "C1", "Cz", "C2", "C4",  # primary motor
    "CP3", "CPz", "CP4",  # somatosensory
    "T7", "T8",  # lateral
    "Pz",  # parietal context
)

TARGET_SFREQ: float = 125.0  # Hz, OpenBCI Cyton+Daisy native sample rate

EPOCH_TMIN: float = -2.0
EPOCH_TMAX: float = 4.0
BASELINE_TMIN: float = -1.5
BASELINE_TMAX: float = -0.5

FILTER_LFREQ: float = 0.5
FILTER_HFREQ: float = 40.0

# Frequency bands of interest (Hz).
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 40.0),
}

# Tasks supported by the production system.
SUPPORTED_TASKS: tuple[str, ...] = (
    "left_hand", "right_hand", "feet", "tongue", "rest", "gait",
)


@dataclass(frozen=True)
class HfRepos:
    """Hugging Face Hub repository identifiers (overridable via env)."""

    dataset: str = field(default_factory=lambda: os.getenv(
        "HF_DATASET_REPO", "abachu2005/mes-eeg-processed"))
    model: str = field(default_factory=lambda: os.getenv(
        "HF_MODEL_REPO", "abachu2005/mes-models"))
    sessions: str = field(default_factory=lambda: os.getenv(
        "HF_SESSIONS_REPO", "abachu2005/mes-sessions"))
    space: str = field(default_factory=lambda: os.getenv(
        "HF_SPACE_REPO", "abachu2005/mes"))


HF_REPOS = HfRepos()

# Pinned dataset + model commit SHAs (set after first publish).
PINNED_DATASET_SHA: str | None = os.getenv("MES_PINNED_DATASET_SHA")
PINNED_MODEL_SHA: str | None = os.getenv("MES_PINNED_MODEL_SHA")


def cache_root() -> Path:
    """Directory used by MES for downloaded models, MOABB cache, etc.

    Defaults to a directory inside the persistent disk on HF Space ($MES_DATA),
    or ~/.cache/mes locally.
    """
    base = os.getenv("MES_DATA") or os.path.join(
        os.path.expanduser("~"), ".cache", "mes"
    )
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p
