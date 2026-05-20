"""OpenBCI Cyton/Daisy .txt and .csv loader.

OpenBCI GUI exports a text file with a header (lines starting with `%`) and
then comma-separated rows. The first column is a sample index, the next 8 (or
16 with Daisy) columns are EEG channel voltages in microvolts, and trailing
columns are accelerometer + timestamp.
"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mes_core.config import OPENBCI_MONTAGE_16, TARGET_SFREQ

DEFAULT_SFREQ = 125.0


def _parse_header(text: str) -> dict[str, str]:
    """Parse OpenBCI header lines starting with `%`."""
    headers: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("%"):
            break
        body = line.lstrip("%").strip()
        if "=" in body:
            k, _, v = body.partition("=")
            headers[k.strip().lower()] = v.strip()
        elif ":" in body:
            k, _, v = body.partition(":")
            headers[k.strip().lower()] = v.strip()
    return headers


def _infer_sfreq(headers: dict[str, str]) -> float:
    raw = headers.get("sample rate") or headers.get("sampling rate") or ""
    m = re.search(r"(\d+(?:\.\d+)?)", raw)
    if m:
        return float(m.group(1))
    return DEFAULT_SFREQ


def _infer_n_eeg(headers: dict[str, str], n_data_cols: int) -> int:
    """Heuristically determine the number of EEG channels."""
    raw = headers.get("number of channels") or ""
    m = re.search(r"(\d+)", raw)
    if m:
        n = int(m.group(1))
        if 1 <= n <= n_data_cols:
            return n
    # Assume the file is sample_idx + N EEG + 3 accel + timestamp(s).
    # If we can shave off 4-5 trailing cols and land on 8 or 16, that's likely.
    for candidate in (16, 8, 4):
        if n_data_cols - candidate >= 4:
            return candidate
    return max(1, n_data_cols - 4)


def load_openbci_txt(
    path: str | os.PathLike[str],
    *,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
    preload: bool = True,
    verbose: str = "ERROR",
) -> Any:
    """Load an OpenBCI Cyton/Daisy raw text export as an mne.io.Raw.

    Parameters
    ----------
    path
        Path to the .txt or .csv exported by the OpenBCI GUI.
    sfreq
        Override the sample rate if the header is missing/wrong.
    ch_names
        Override channel names. Defaults to the 16-channel OpenBCI sensorimotor
        montage when 16 EEG columns are present; otherwise EEG001..EEG00N.
    """
    import mne

    p = Path(path)
    text = p.read_text(errors="ignore")
    headers = _parse_header(text)
    sf = sfreq if sfreq is not None else _infer_sfreq(headers)

    df = pd.read_csv(
        io.StringIO(text),
        comment="%",
        header=None,
        skip_blank_lines=True,
        engine="python",
    )
    if df.empty:
        raise ValueError(f"OpenBCI file {p} contains no data rows")

    # Strip the first column if it looks like a sample index.
    first_col = df.iloc[:, 0]
    if pd.api.types.is_numeric_dtype(first_col) and np.allclose(
        np.diff(first_col.values[:50]).astype(int) % 256,
        np.full(49, 1),
    ):
        df = df.iloc[:, 1:]

    n_data_cols = df.shape[1]
    n_eeg = _infer_n_eeg(headers, n_data_cols)
    eeg = df.iloc[:, :n_eeg].to_numpy(dtype=np.float64).T  # (n_eeg, n_samples)

    # OpenBCI exports microvolts -> MNE expects volts.
    eeg_volts = eeg * 1e-6

    if ch_names is None:
        if n_eeg == 16:
            ch_names = list(OPENBCI_MONTAGE_16)
        elif n_eeg == 8:
            ch_names = ["Fp1", "Fp2", "C3", "C4", "T7", "T8", "O1", "O2"]
        else:
            ch_names = [f"EEG{i + 1:03d}" for i in range(n_eeg)]

    info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types="eeg")
    try:
        info.set_montage("standard_1020", on_missing="ignore", verbose=verbose)
    except Exception:
        pass
    raw = mne.io.RawArray(eeg_volts, info, verbose=verbose)
    if not preload:
        raw.load_data(verbose=verbose)
    return raw


def expected_openbci_sfreq() -> float:
    return TARGET_SFREQ
