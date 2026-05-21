"""Rehab proxy index tests."""

from __future__ import annotations

import numpy as np

from mes_core.scoring.rehab_proxy import compute_rehab_proxy, paretic_hand_for_side


def test_paretic_hand_mapping() -> None:
    assert paretic_hand_for_side("left") == "left_hand"
    assert paretic_hand_for_side("Right") == "right_hand"


def test_rehab_proxy_weights_clinical() -> None:
    mes = np.array([80.0, 75.0, 10.0, 12.0])
    labels = ["right_hand", "right_hand", "break", "break"]
    r = compute_rehab_proxy(
        mes,
        labels,
        paralysis_side="right",
        nihss=4.0,
        mbi=80.0,
    )
    assert r.n_paretic_trials == 2
    assert r.mes_paretic_mean == 77.5
    assert r.capacity_weight < 1.0
    assert r.rehab_proxy_index < r.mes_paretic_mean
