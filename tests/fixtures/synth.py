"""Synthetic EEG fixture generators.

Produces deterministic, physiologically-plausible signals for testing the
full preprocessing -> feature -> scoring pipeline without needing real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mes_core.config import OPENBCI_MONTAGE_16

RNG_SEED_DEFAULT = 1729


@dataclass
class SyntheticSpec:
    """Describes a synthetic EEG recording."""

    n_channels: int = 16
    sfreq: float = 125.0
    duration_s: float = 30.0
    n_trials: int = 10
    trial_len_s: float = 6.0  # 2 s pre, 4 s post stimulus
    pink_noise_std: float = 1.0
    mu_amplitude: float = 8.0  # base mu power amplitude in synthetic units
    beta_amplitude: float = 4.0
    erd_strength: float = 0.6  # 0 = no ERD, 1 = full suppression of mu/beta during task
    contra_channel_idx: int = 5  # default C3 in OPENBCI_MONTAGE_16
    ipsi_channel_idx: int = 9    # default C4
    seed: int = RNG_SEED_DEFAULT


def _pink_noise(n_samples: int, n_channels: int, rng: np.random.Generator) -> np.ndarray:
    """Generate 1/f-shaped noise via FFT shaping."""
    white = rng.standard_normal((n_channels, n_samples))
    fft = np.fft.rfft(white, axis=-1)
    freqs = np.fft.rfftfreq(n_samples, d=1.0)
    scale = 1.0 / np.sqrt(np.maximum(freqs, 1e-3))
    fft *= scale
    out: np.ndarray = np.fft.irfft(fft, n=n_samples, axis=-1)
    return out


def make_continuous(spec: SyntheticSpec) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate a continuous synthetic recording with embedded ERD trials.

    Returns
    -------
    data : (n_channels, n_samples) float32 in microvolts-ish units.
    events : (n_trials, 3) array of (sample, 0, event_id).  event_id=1.
    ch_names : list of channel labels (uses OPENBCI_MONTAGE_16 by default).
    """
    rng = np.random.default_rng(spec.seed)
    n_samples = int(round(spec.duration_s * spec.sfreq))
    t = np.arange(n_samples) / spec.sfreq

    data = spec.pink_noise_std * _pink_noise(n_samples, spec.n_channels, rng)

    # Add baseline mu and beta to every channel
    for f, amp in [(10.5, spec.mu_amplitude), (20.0, spec.beta_amplitude)]:
        phase = rng.uniform(0, 2 * np.pi, size=spec.n_channels)
        for c in range(spec.n_channels):
            data[c] += amp * np.sin(2 * np.pi * f * t + phase[c])

    # Schedule trials uniformly across the recording.
    if spec.n_trials > 0:
        trial_samples = int(round(spec.trial_len_s * spec.sfreq))
        gap = max(1, (n_samples - trial_samples) // (spec.n_trials + 1))
        events_list: list[tuple[int, int, int]] = []
        for i in range(spec.n_trials):
            onset = (i + 1) * gap
            events_list.append((onset, 0, 1))
            # Suppress mu+beta on the contralateral channel in the post-stim window.
            post_start = onset
            post_end = min(n_samples, onset + int(2.0 * spec.sfreq))
            data[spec.contra_channel_idx, post_start:post_end] *= (1.0 - spec.erd_strength)
        events = np.array(events_list, dtype=int)
    else:
        events = np.zeros((0, 3), dtype=int)

    n_ch = spec.n_channels
    if n_ch == len(OPENBCI_MONTAGE_16):
        ch_names = list(OPENBCI_MONTAGE_16)
    else:
        ch_names = [f"EEG{i:03d}" for i in range(n_ch)]
    return data.astype(np.float32), events, ch_names


def make_mne_raw(spec: SyntheticSpec | None = None):  # type: ignore[no-untyped-def]
    """Convenience: return an mne.io.Raw with synthetic data."""
    import mne

    if spec is None:
        spec = SyntheticSpec()
    data, events, ch_names = make_continuous(spec)
    info = mne.create_info(ch_names=ch_names, sfreq=spec.sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")  # mne expects volts
    raw.set_annotations(
        mne.Annotations(
            onset=events[:, 0] / spec.sfreq,
            duration=np.full(len(events), spec.trial_len_s),
            description=["task"] * len(events),
        )
    )
    return raw, events
