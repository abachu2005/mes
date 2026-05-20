"""Hypothesis property-based tests for signal-processing utilities."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mes_core.features.bandpower import band_power, erd_percent


@settings(deadline=None, max_examples=30)
@given(
    n_channels=st.integers(min_value=1, max_value=8),
    n_samples=st.integers(min_value=64, max_value=512),
    sfreq=st.floats(min_value=100.0, max_value=500.0, allow_nan=False),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_band_power_shape_and_nonneg(n_channels, n_samples, sfreq, seed) -> None:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_channels, n_samples))
    p = band_power(data, sfreq, (4.0, 12.0))
    assert p.shape == (n_channels,)
    assert np.all(p >= 0)


@settings(deadline=None, max_examples=30)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
)
def test_erd_is_bounded_below_by_minus_inf(seed, scale) -> None:
    rng = np.random.default_rng(seed)
    sfreq = 250.0
    n = 256
    base = rng.standard_normal((4, n))
    task = scale * base
    erd = erd_percent(task, base, sfreq, (8.0, 13.0))
    assert np.all(np.isfinite(erd))
    assert np.all(erd <= 100.0)  # power can't go below zero
