"""Rest vs task epoch splitting."""

from __future__ import annotations

import numpy as np

from mes_core.scoring.rest import rest_mask_protocol, split_rest_and_task_epochs


def test_implicit_rest_block_detected() -> None:
    rng = np.random.default_rng(0)
    calm = rng.standard_normal((4, 16, 500)) * 0.001
    active = rng.standard_normal((12, 16, 500)) * 0.2
    data = np.concatenate([calm, active], axis=0)
    rest_idx, task_idx, kind = split_rest_and_task_epochs(data, sfreq=125.0)
    assert kind == "implicit_rest_block"
    assert len(rest_idx) >= 1
    assert len(task_idx) == 12


def test_rest_mask_protocol_first_windows() -> None:
    # 180 s recording @ 3 s step -> 60 epochs; first 20 start before 60 s
    n = 60
    mask = rest_mask_protocol(n, sfreq=125.0, rest_seconds=60.0, window_s=6.0, step_s=3.0)
    assert mask.shape == (n,)
    assert mask.sum() == 20
    assert mask[0] and mask[19]
    assert not mask[20]
