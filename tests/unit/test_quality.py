"""Quality gate tests."""

from __future__ import annotations

import numpy as np

from mes_core.config import OPENBCI_MONTAGE_16
from mes_core.quality import assess_session, reliability_tier


def test_flat_epoch_fails_quality() -> None:
    ch = list(OPENBCI_MONTAGE_16)
    data = np.zeros((3, len(ch), 500))
    sq, _mask = assess_session(data, ch, sfreq=125.0)
    assert sq.n_usable == 0
    assert not sq.ok


def test_reasonable_epoch_passes() -> None:
    rng = np.random.default_rng(1)
    ch = list(OPENBCI_MONTAGE_16)
    data = rng.standard_normal((8, len(ch), 750)) * 1e-5
    sq, _mask = assess_session(data, ch, sfreq=125.0)
    assert sq.fraction_usable >= 0.5
    tier = reliability_tier(sq, baseline_kind="subject_rest")
    assert tier in ("High", "Medium", "Low")
