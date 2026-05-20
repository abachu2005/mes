"""I/O loader tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mes_core.io import load_eeg, load_openbci_txt


def test_load_eeg_rejects_unknown_extension(tmp_path):
    bad = tmp_path / "weird.xyz"
    bad.write_text("garbage")
    with pytest.raises(ValueError, match="Unsupported"):
        load_eeg(bad)


def test_load_eeg_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eeg(tmp_path / "nope.edf")


def test_load_openbci_txt_roundtrip(tmp_path):
    sfreq = 125.0
    n = 500
    t = np.arange(n) / sfreq
    eeg = np.vstack(
        [np.sin(2 * np.pi * (8 + i) * t) for i in range(16)]
    ).T * 10.0  # microvolts
    accel = np.zeros((n, 3))
    sample_idx = (np.arange(n) % 256).reshape(-1, 1)

    arr = np.hstack([sample_idx, eeg, accel])
    df = pd.DataFrame(arr)
    txt = tmp_path / "rec.txt"
    header = (
        "%OpenBCI Raw EXG Data\n"
        "%Number of channels = 16\n"
        "%Sample Rate = 125 Hz\n"
        "%Board = Cyton + Daisy\n"
    )
    with txt.open("w") as f:
        f.write(header)
        df.to_csv(f, index=False, header=False)

    raw = load_openbci_txt(txt)
    assert raw.info["sfreq"] == pytest.approx(125.0)
    assert len(raw.info["ch_names"]) == 16
    # MNE stores volts; original was microvolts so peak amplitude ~10e-6 V
    assert raw.get_data().max() < 1e-3
