"""Generic file-format-aware EEG loader.

Dispatches based on file extension to the right MNE reader (EDF, BDF, GDF, FIF)
or our custom OpenBCI .txt loader.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_eeg(path: str | os.PathLike[str], *, preload: bool = True, verbose: str = "ERROR") -> Any:
    """Load an EEG recording from disk and return an mne.io.Raw object.

    Supported extensions: .edf, .bdf, .gdf, .fif, .vhdr (BrainVision),
    .set (EEGLAB), .txt/.csv (OpenBCI).

    Parameters
    ----------
    path
        Path to the input file.
    preload
        If True, data is loaded into memory immediately. Required for most
        downstream operations.
    verbose
        MNE verbosity level.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If the file extension is unsupported.
    """
    import mne

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    ext = p.suffix.lower()

    if ext == ".edf":
        return mne.io.read_raw_edf(p, preload=preload, verbose=verbose)
    if ext == ".bdf":
        return mne.io.read_raw_bdf(p, preload=preload, verbose=verbose)
    if ext == ".gdf":
        return mne.io.read_raw_gdf(p, preload=preload, verbose=verbose)
    if ext == ".fif":
        return mne.io.read_raw_fif(p, preload=preload, verbose=verbose)
    if ext == ".vhdr":
        return mne.io.read_raw_brainvision(p, preload=preload, verbose=verbose)
    if ext == ".set":
        return mne.io.read_raw_eeglab(p, preload=preload, verbose=verbose)
    if ext in {".txt", ".csv"}:
        from mes_core.io.openbci import load_openbci_txt
        return load_openbci_txt(p, preload=preload, verbose=verbose)

    raise ValueError(f"Unsupported EEG file extension: {ext}")
