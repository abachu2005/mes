"""Tests for the topomap payload + HTML report rendering."""

from __future__ import annotations

import numpy as np

from mes_core.config import OPENBCI_MONTAGE_16
from mes_core.viz.report import ReportContext, build_session_report_html
from mes_core.viz.topomap import scalp_topomap_payload


def test_topomap_payload_picks_known_positions() -> None:
    values = np.linspace(-1, 1, len(OPENBCI_MONTAGE_16))
    payload = scalp_topomap_payload(values, list(OPENBCI_MONTAGE_16))
    names = {p["channel"] for p in payload["points"]}
    assert "C3" in names and "Cz" in names and "Pz" in names
    assert payload["vmin"] <= payload["vmax"]


def test_html_report_renders_with_mock_context() -> None:
    ctx = ReportContext(
        participant_code="P001",
        session_id="S-0001",
        task="right_hand",
        target_limb="Right hand",
        headset="OpenBCI Cyton+Daisy",
        mes_mean=62.4,
        mes_median=60.0,
        mes_std=12.5,
        n_trials=15,
        mes_per_trial=list(np.linspace(40, 80, 15)),
        erd_topomap=scalp_topomap_payload(
            np.linspace(-30, 30, len(OPENBCI_MONTAGE_16)), list(OPENBCI_MONTAGE_16)
        ),
        lateralization=0.43,
        model_sha="abc1234567890def",
        created_at="2026-05-19 13:00 UTC",
        notes="Pilot session.",
    )
    html = build_session_report_html(ctx)
    assert "MES" in html
    assert "P001" in html
    assert "data:image/png;base64," in html
