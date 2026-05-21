"""Bundled MES weights artifacts."""

from __future__ import annotations

from mes_core.artifacts import load_mes_weights, load_population_baseline


def test_load_fitted_weights_not_defaults() -> None:
    w = load_mes_weights("right_hand")
    assert w.w_pmodel > 0.5
    assert abs(w.intercept) > 0.1


def test_population_baseline_has_four_features() -> None:
    bl = load_population_baseline("right_hand")
    assert bl.mean.shape == (4,)
    assert bl.std.shape == (4,)
