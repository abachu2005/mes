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


def contra_ipsi_for_stroke(
    task: str,
    paralysis_side: str | None = None,
) -> tuple[list[str], list[str]]:
    """Contra/ipsi channels for stroke MI, optionally adjusted for hemiplegia side.

    When the cued hand is the **non-paretic** hand, swap contra/ipsi so lateralization
    index reflects engagement of the hemisphere that would drive the paretic limb.
    """
    contra, ipsi = default_contra_ipsi_for_task(task)
    if not paralysis_side:
        return contra, ipsi
    p = paralysis_side.lower().strip()
    paretic_left = p in ("left", "l", "left_hemiplegia", "left hemiplegia", "left hemiparesis")
    paretic_right = p in ("right", "r", "right_hemiplegia", "right hemiplegia", "right hemiparesis")
    task_right = "right" in task.lower()
    task_left = "left" in task.lower()
    if paretic_left and task_right:
        return ipsi, contra
    if paretic_right and task_left:
        return ipsi, contra
    return contra, ipsi
