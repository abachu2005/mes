"""Sanity tests on the synthetic EEG fixture itself."""

from __future__ import annotations

import numpy as np

from tests.fixtures.synth import SyntheticSpec, make_continuous


def test_make_continuous_shape_and_determinism() -> None:
    spec = SyntheticSpec(seed=42)
    a, events_a, names_a = make_continuous(spec)
    b, events_b, names_b = make_continuous(spec)

    assert a.shape == (spec.n_channels, int(spec.duration_s * spec.sfreq))
    assert events_a.shape == (spec.n_trials, 3)
    assert np.array_equal(a, b)  # determinism
    assert np.array_equal(events_a, events_b)
    assert names_a == names_b


def test_make_continuous_erd_is_visible() -> None:
    """Power on the contra channel should drop after each trial onset."""
    spec = SyntheticSpec(seed=1, erd_strength=0.8, n_trials=5, duration_s=30.0)
    data, events, _ = make_continuous(spec)

    sfreq = spec.sfreq
    contra = data[spec.contra_channel_idx]

    # Compare RMS over [-1, 0) pre vs (0, 1] post.
    onset = events[0, 0]
    pre = contra[max(0, onset - int(sfreq)): onset]
    post = contra[onset: onset + int(sfreq)]
    assert post.var() < pre.var(), "Expected post-stimulus variance to drop on contra channel"
