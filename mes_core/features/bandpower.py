"""Spectral band power and Event-Related Desynchronization (ERD%).

ERD% = (P_baseline - P_task) / P_baseline * 100,
positive values = power reduction = desynchronization = motor engagement.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.signal import welch

from mes_core.config import BANDS


def band_power(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    *,
    nperseg: int | None = None,
) -> np.ndarray:
    """Compute mean power in `band` for each (..., channel, time) array.

    Returns array with the last (time) axis collapsed: shape (..., n_channels).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim < 2:
        raise ValueError("data must have at least 2 dims (channels, time)")
    n_times = data.shape[-1]
    if nperseg is None:
        nperseg = min(n_times, round(sfreq))
    nperseg = max(8, min(nperseg, n_times))

    freqs, psd = welch(
        data,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=-1,
        detrend="constant",
    )
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return np.zeros(data.shape[:-1])
    return psd[..., mask].mean(axis=-1)


def all_band_powers(
    data: np.ndarray,
    sfreq: float,
    bands: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    bands = bands or BANDS
    return {name: band_power(data, sfreq, b) for name, b in bands.items()}


def erd_percent(
    task_data: np.ndarray,
    baseline_data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
) -> np.ndarray:
    """Event-Related Desynchronization (percent).

    Positive -> power drop during task vs baseline -> motor engagement.
    """
    p_task = band_power(task_data, sfreq, band)
    p_base = band_power(baseline_data, sfreq, band)
    return 100.0 * (p_base - p_task) / np.maximum(p_base, 1e-12)
