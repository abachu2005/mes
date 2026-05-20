"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.synth import SyntheticSpec, make_continuous, make_mne_raw


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1729)


@pytest.fixture
def synth_spec() -> SyntheticSpec:
    return SyntheticSpec()


@pytest.fixture
def synth_data(synth_spec: SyntheticSpec):
    return make_continuous(synth_spec)


@pytest.fixture
def synth_raw(synth_spec: SyntheticSpec):
    return make_mne_raw(synth_spec)
