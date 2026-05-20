"""Lateralization index.

LI = (ERD_contra - ERD_ipsi) / (|ERD_contra| + |ERD_ipsi| + eps)
"""

from __future__ import annotations

import numpy as np


def lateralization_index(
    erd_per_channel: np.ndarray,
    ch_names: list[str],
    *,
    contra_channels: list[str],
    ipsi_channels: list[str],
) -> float | np.ndarray:
    """Compute the lateralization index from per-channel ERD values.

    `erd_per_channel` can be shape (n_channels,) or (n_trials, n_channels);
    we average across the requested channel groups.
    """
    arr = np.asarray(erd_per_channel, dtype=float)
    idx_contra = [ch_names.index(c) for c in contra_channels if c in ch_names]
    idx_ipsi = [ch_names.index(c) for c in ipsi_channels if c in ch_names]
    if not idx_contra or not idx_ipsi:
        return float("nan") if arr.ndim == 1 else np.full(arr.shape[0], np.nan)

    contra = arr[..., idx_contra].mean(axis=-1)
    ipsi = arr[..., idx_ipsi].mean(axis=-1)
    denom = np.abs(contra) + np.abs(ipsi) + 1e-9
    return (contra - ipsi) / denom


def default_contra_ipsi_for_task(task: str) -> tuple[list[str], list[str]]:
    """Return (contra, ipsi) channel lists for a target task.

    For "right_hand": contra = left hemisphere (C3, FC3, CP3); ipsi = C4/FC4/CP4.
    Convention: a participant moving their right hand activates the LEFT motor cortex.
    """
    left = ["C3", "FC3", "CP3", "C1"]
    right = ["C4", "FC4", "CP4", "C2"]
    midline = ["Cz", "FCz", "CPz"]

    t = task.lower()
    if "right" in t:
        return left, right
    if "left" in t:
        return right, left
    if "foot" in t or "feet" in t or "gait" in t:
        return midline, midline
    return midline, midline
