"""Preprocessing pipeline unit tests."""

from __future__ import annotations

import numpy as np

from mes_core.config import OPENBCI_MONTAGE_16, TARGET_SFREQ
from mes_core.preprocessing import PreprocessConfig, epoch_raw, map_to_openbci_16, preprocess_raw


def test_preprocess_returns_target_sfreq_and_channels(synth_raw) -> None:
    raw, _ = synth_raw
    cfg = PreprocessConfig(do_ica=False, autoreject_epochs=False)
    out = preprocess_raw(raw, cfg)
    assert abs(out.info["sfreq"] - TARGET_SFREQ) < 0.5
    assert list(out.info["ch_names"]) == list(OPENBCI_MONTAGE_16)


def test_preprocess_is_idempotent_on_filter_band(synth_raw) -> None:
    raw, _ = synth_raw
    cfg = PreprocessConfig(do_ica=False)
    out1 = preprocess_raw(raw, cfg)
    out2 = preprocess_raw(out1, cfg)
    # Energy is preserved up to filter ringing.
    p1 = (out1.get_data() ** 2).sum()
    p2 = (out2.get_data() ** 2).sum()
    assert abs(p1 - p2) / max(p1, 1e-12) < 0.05


def test_map_to_openbci_with_full_montage_picks_only(synth_raw) -> None:
    raw, _ = synth_raw
    out = map_to_openbci_16(raw)
    assert list(out.info["ch_names"]) == list(OPENBCI_MONTAGE_16)
    assert out.get_data().shape[0] == len(OPENBCI_MONTAGE_16)


def test_map_to_openbci_with_missing_channels_interpolates() -> None:
    import mne

    present = list(OPENBCI_MONTAGE_16[:12])  # drop the last 4
    n_samples = 500
    data = np.random.default_rng(0).standard_normal((len(present), n_samples)) * 1e-6
    info = mne.create_info(ch_names=present, sfreq=125.0, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    out = map_to_openbci_16(raw)
    assert list(out.info["ch_names"]) == list(OPENBCI_MONTAGE_16)


def test_epoch_raw_runs_on_synthetic(synth_raw) -> None:
    raw, _ = synth_raw
    cfg = PreprocessConfig(do_ica=False)
    raw_pp = preprocess_raw(raw, cfg)
    epochs = epoch_raw(raw_pp, tmin=-2.0, tmax=4.0, baseline=(-1.5, -0.5))
    assert epochs.get_data().ndim == 3
    assert epochs.get_data().shape[1] == len(OPENBCI_MONTAGE_16)
