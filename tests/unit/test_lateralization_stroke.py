"""Stroke hemiplegia contra/ipsi mapping."""

from mes_core.features.lateralization import contra_ipsi_for_stroke, default_contra_ipsi_for_task


def test_stroke_swap_for_non_paretic_right_hand() -> None:
    contra, ipsi = contra_ipsi_for_stroke("right_hand", "left")
    exp_c, exp_i = default_contra_ipsi_for_task("right_hand")
    assert contra == exp_i
    assert ipsi == exp_c


def test_stroke_no_swap_paretic_right_hand() -> None:
    contra, ipsi = contra_ipsi_for_stroke("right_hand", "right")
    assert contra == default_contra_ipsi_for_task("right_hand")[0]
