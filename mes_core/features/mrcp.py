"""Movement-Related Cortical Potential (MRCP) features.

MRCP is a slow negative deflection (0.1-3 Hz) over the sensorimotor cortex,
peaking around movement onset. We extract:
- amplitude: signed peak in the (-0.5, +0.5)s window
- slope: linear slope across (-1.5, +0.5)s
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _butter_bandpass(low: float, high: float, sfreq: float, order: int = 4):
    nyq = 0.5 * sfreq
    if high >= nyq:
        high = nyq * 0.99
    if low <= 0:
        low = 0.05
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def mrcp_features(
    epoch_data: np.ndarray,
    sfreq: float,
    *,
    tmin: float = -2.0,
    onset_t: float = 0.0,
    onset_window: tuple[float, float] = (-0.5, 0.5),
    slope_window: tuple[float, float] = (-1.5, 0.5),
    low: float = 0.1,
    high: float = 3.0,
) -> dict[str, np.ndarray]:
    """Compute MRCP amplitude + slope per (epoch, channel).

    Parameters
    ----------
    epoch_data : (n_epochs, n_channels, n_times) or (n_channels, n_times)
    sfreq : float
    tmin : float, time of the first sample (s, relative to onset_t)
    """
    data = np.asarray(epoch_data, dtype=float)
    single = data.ndim == 2
    if single:
        data = data[None, ...]

    sos = _butter_bandpass(low, high, sfreq)
    filt = sosfiltfilt(sos, data, axis=-1)

    n_times = data.shape[-1]
    times = tmin + np.arange(n_times) / sfreq

    a0, a1 = onset_window
    am = (times >= a0) & (times <= a1)
    if am.any():
        seg = filt[..., am]
        max_abs_idx = np.argmax(np.abs(seg), axis=-1)
        amp = np.take_along_axis(seg, max_abs_idx[..., None], axis=-1).squeeze(-1)
    else:
        amp = np.zeros(data.shape[:-1])

    s0, s1 = slope_window
    sm = (times >= s0) & (times <= s1)
    if sm.sum() >= 2:
        t_seg = times[sm]
        seg = filt[..., sm]
        x_mean = t_seg.mean()
        y_mean = seg.mean(axis=-1, keepdims=True)
        num = ((t_seg - x_mean) * (seg - y_mean)).sum(axis=-1)
        denom = ((t_seg - x_mean) ** 2).sum()
        slope = num / max(denom, 1e-12)
    else:
        slope = np.zeros(data.shape[:-1])

    if single:
        amp = amp.squeeze(0)
        slope = slope.squeeze(0)
    return {"amplitude": amp, "slope": slope}
