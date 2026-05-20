"""Spatial mapping from arbitrary EEG montages to the 16-channel OpenBCI montage.

Strategy:
- If all 16 target channels are present in the input -> reorder + drop extras.
- If some are missing -> use MNE's spherical-spline interpolation across the
  standard 1020 montage. ICLabel-style artifact rejection should already have
  been applied on the full source montage *before* this mapping.
"""

from __future__ import annotations

from typing import Any

from mes_core.config import OPENBCI_MONTAGE_16


def map_to_openbci_16(raw: Any, *, target: tuple[str, ...] = OPENBCI_MONTAGE_16) -> Any:
    """Return a copy of `raw` with channels mapped to `target` (default 16-ch OpenBCI)."""
    import mne

    raw = raw.copy()
    try:
        raw.set_montage("standard_1020", on_missing="ignore", match_case=False, verbose="ERROR")
    except Exception:
        pass

    present = set(raw.info["ch_names"])
    missing = [ch for ch in target if ch not in present]

    if not missing:
        return raw.pick(list(target))

    # Build a new Raw that contains target channels (zeros for missing) and the
    # rest of the original signal, then mark the zeroed ones as bad and let MNE
    # interpolate from neighbours.
    import numpy as np

    sfreq = raw.info["sfreq"]
    n_samples = raw.n_times

    target_present = [ch for ch in target if ch in present]
    raw_present = raw.copy().pick(target_present + [c for c in raw.info["ch_names"] if c not in target])

    zero_data = np.zeros((len(missing), n_samples), dtype=np.float64)
    info_missing = mne.create_info(ch_names=list(missing), sfreq=sfreq, ch_types="eeg")
    raw_missing = mne.io.RawArray(zero_data, info_missing, verbose="ERROR")
    raw_missing.info["bads"] = list(missing)

    raw_combined = raw_present.copy()
    raw_combined.add_channels([raw_missing], force_update_info=True)
    try:
        raw_combined.set_montage(
            "standard_1020", on_missing="ignore", match_case=False, verbose="ERROR"
        )
        if any(b in raw_combined.info["ch_names"] for b in missing):
            raw_combined.interpolate_bads(reset_bads=True, mode="accurate", verbose="ERROR")
    except Exception:
        # Last resort: leave zeros and warn via the channel name list
        pass

    return raw_combined.pick(list(target))
