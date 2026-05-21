"""Protocol upload scoring: 60 s rest block + sliding windows."""

from __future__ import annotations

from pathlib import Path

import pytest

from mes_core.pipeline import score_recording


@pytest.fixture
def protocol_txt(tmp_path: Path) -> Path:
    from scripts.generate_protocol_test_file import build_protocol_recording, write_openbci_txt

    out = tmp_path / "protocol.txt"
    data = build_protocol_recording(n_trials=8)
    write_openbci_txt(out, data)
    return out


def test_protocol_rest_block_splits_epochs(protocol_txt: Path) -> None:
    scored = score_recording(protocol_txt, had_rest_block=True, require_quality=False)
    assert scored.n_rest_epochs >= 10
    assert scored.n_task_epochs >= 10
    assert scored.baseline_kind == "subject_rest"
    assert scored.reliability in ("High", "Medium", "Low")


def test_protocol_rest_improves_baseline_kind(protocol_txt: Path) -> None:
    without = score_recording(protocol_txt, had_rest_block=False, require_quality=False)
    with_rest = score_recording(protocol_txt, had_rest_block=True, require_quality=False)
    assert with_rest.n_rest_epochs > without.n_rest_epochs
    assert with_rest.baseline_kind == "subject_rest"
