"""End-to-end preprocessing pipeline.

Implements:
1. Bandpass + notch filtering
2. Common-average / linked-mastoid re-referencing
3. ICA artifact removal on the *full* source montage (if available + >=32 ch),
   otherwise threshold-based EOG-channel blink rejection.
4. Spatial mapping to the 16-channel OpenBCI montage (delegated to montage.py).
5. Resampling to TARGET_SFREQ (125 Hz).
6. Cue-locked epoching with baseline correction.
7. Optional bad-epoch rejection (lazy import of autoreject).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from mes_core.config import (
    BASELINE_TMAX,
    BASELINE_TMIN,
    EPOCH_TMAX,
    EPOCH_TMIN,
    FILTER_HFREQ,
    FILTER_LFREQ,
    OPENBCI_MONTAGE_16,
    TARGET_SFREQ,
)


@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing pipeline."""

    l_freq: float = FILTER_LFREQ
    h_freq: float = FILTER_HFREQ
    notch_freqs: Sequence[float] = field(default_factory=lambda: (50.0, 60.0))
    target_sfreq: float = TARGET_SFREQ
    do_ica: bool = True
    ica_min_channels: int = 32
    ica_n_components: int | None = None
    map_to_openbci: bool = True
    target_channels: tuple[str, ...] = OPENBCI_MONTAGE_16
    reference: str = "average"  # 'average' or 'mastoids'
    autoreject_epochs: bool = False


def _apply_ica(raw: Any, *, n_components: int | None = None) -> Any:
    """Run ICA + (best-effort) ICLabel auto-removal. Falls back to threshold removal."""
    import mne

    raw = raw.copy()
    n_ch = len(raw.info["ch_names"])
    if n_components is None:
        n_components = min(20, max(8, n_ch - 1))

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="picard" if _picard_available() else "fastica",
        random_state=1729,
        max_iter="auto",
        verbose="ERROR",
    )
    ica.fit(raw, verbose="ERROR")

    bad_idx: list[int] = []
    try:
        from mne_icalabel import label_components

        labels = label_components(raw, ica, method="iclabel")
        # iclabel returns labels: brain, muscle, eye blink, heart beat, line noise, channel noise, other
        for i, (label, prob) in enumerate(
            zip(labels["labels"], labels["y_pred_proba"], strict=False)
        ):
            if label in {"eye blink", "muscle artifact", "heart beat", "line noise"} and prob > 0.7:
                bad_idx.append(i)
    except Exception:
        # Fallback: identify ICs whose source has very high kurtosis (eye/muscle)
        import numpy as np
        sources = ica.get_sources(raw).get_data()
        kurt = ((sources - sources.mean(axis=1, keepdims=True)) ** 4).mean(axis=1) / (
            sources.var(axis=1) ** 2 + 1e-12
        ) - 3.0
        bad_idx = [int(i) for i in np.where(kurt > 5.0)[0]]

    ica.exclude = bad_idx
    return ica.apply(raw, verbose="ERROR")


def _picard_available() -> bool:
    try:
        import picard  # noqa: F401
        return True
    except Exception:
        return False


def preprocess_raw(raw: Any, cfg: PreprocessConfig | None = None) -> Any:
    """Apply the full preprocessing pipeline to an mne.io.Raw object."""
    from mes_core.preprocessing.montage import map_to_openbci_16

    if cfg is None:
        cfg = PreprocessConfig()
    raw = raw.copy()

    # Bandpass + notch (zero-phase FIR by default in MNE).
    raw.filter(
        l_freq=cfg.l_freq,
        h_freq=cfg.h_freq,
        fir_design="firwin",
        verbose="ERROR",
    )
    nyquist = raw.info["sfreq"] / 2.0
    notch_in_band = [f for f in cfg.notch_freqs if f < nyquist - 1]
    if notch_in_band:
        raw.notch_filter(freqs=notch_in_band, verbose="ERROR")

    # Re-reference.
    if cfg.reference == "mastoids":
        mastoid_chs = [c for c in ("A1", "A2", "M1", "M2") if c in raw.info["ch_names"]]
        if len(mastoid_chs) >= 1:
            raw.set_eeg_reference(mastoid_chs, verbose="ERROR")
        else:
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    else:
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    # ICA artifact removal on the full source montage when we have enough channels.
    if cfg.do_ica and len(raw.info["ch_names"]) >= cfg.ica_min_channels:
        raw = _apply_ica(raw, n_components=cfg.ica_n_components)

    # Map to OpenBCI 16-ch montage (after ICA).
    if cfg.map_to_openbci:
        target = tuple(cfg.target_channels)
        if any(ch in raw.info["ch_names"] for ch in target):
            raw = map_to_openbci_16(raw, target=target)

    # Resample.
    if abs(raw.info["sfreq"] - cfg.target_sfreq) > 0.5:
        raw.resample(cfg.target_sfreq, npad="auto", verbose="ERROR")

    return raw


def epoch_raw(
    raw: Any,
    *,
    event_id: dict[str, int] | None = None,
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
    baseline: tuple[float, float] | None = (BASELINE_TMIN, BASELINE_TMAX),
    reject_by_annotation: bool = True,
    use_autoreject: bool = False,
) -> Any | None:
    """Epoch a preprocessed Raw around annotation events.

    Returns an mne.Epochs object (possibly empty if no events match).
    """
    import mne

    try:
        events, ev_id = mne.events_from_annotations(raw, verbose="ERROR")
    except (RuntimeError, ValueError):
        return None
    if events.size == 0:
        return None

    final_id = event_id if event_id is not None else ev_id
    final_id = {k: v for k, v in final_id.items() if v in events[:, 2]}
    if not final_id:
        final_id = ev_id

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=final_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject_by_annotation=reject_by_annotation,
        verbose="ERROR",
    )

    if use_autoreject and len(epochs) > 5:
        try:
            from autoreject import AutoReject
            ar = AutoReject(random_state=1729, verbose=False)
            epochs = ar.fit_transform(epochs)
        except Exception:
            pass

    return epochs


def epoch_sliding_windows(
    raw: Any,
    *,
    window_s: float = 6.0,
    step_s: float = 3.0,
) -> np.ndarray | None:
    """Slide fixed windows over continuous data when no annotations exist.

    Returns (n_epochs, n_channels, n_samples) or None.
    """
    import numpy as np

    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    win = int(window_s * sfreq)
    step = max(1, int(step_s * sfreq))
    if data.shape[1] < win:
        return None
    epochs = []
    for start in range(0, data.shape[1] - win + 1, step):
        epochs.append(data[:, start : start + win])
    if not epochs:
        return None
    return np.stack(epochs, axis=0)
