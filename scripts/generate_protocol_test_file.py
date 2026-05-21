#!/usr/bin/env python3
"""Generate an OpenBCI-style .txt with 60 s rest + task blocks (mu ERD on C3).

Upload with had_rest_block=true for protocol-aligned rest/task scoring.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fixtures.synth import SyntheticSpec, _pink_noise, make_continuous

SFREQ = 125.0
REST_S = 60.0


def build_protocol_recording(
    *,
    rest_s: float = REST_S,
    n_trials: int = 12,
    sfreq: float = SFREQ,
    seed: int = 42,
) -> np.ndarray:
    """60 s calm rest, then MI trials with contralateral ERD (C3)."""
    rng = np.random.default_rng(seed)
    n_rest = int(rest_s * sfreq)
    t_rest = np.arange(n_rest) / sfreq
    n_ch = 16
    rest_block = 1.0 * _pink_noise(n_rest, n_ch, rng)
    for f, amp in [(10.5, 8.0), (20.0, 4.0)]:
        phase = rng.uniform(0, 2 * np.pi, size=n_ch)
        for c in range(n_ch):
            rest_block[c] += amp * np.sin(2 * np.pi * f * t_rest + phase[c])

    task_spec = SyntheticSpec(
        duration_s=max(120.0, n_trials * 15.0),
        n_trials=n_trials,
        trial_len_s=6.0,
        erd_strength=0.75,
        contra_channel_idx=5,
        ipsi_channel_idx=9,
        seed=seed + 1,
    )
    task_data, _events, _ = make_continuous(task_spec)
    return np.concatenate([rest_block.astype(np.float32), task_data], axis=1)


def write_openbci_txt(path: Path, data: np.ndarray, sfreq: float = SFREQ) -> None:
    """OpenBCI GUI-style export: microvolts, sample index + 16 EEG + 3 accel."""
    n_ch, n_samp = data.shape
    eeg_uv = data.T
    sample_idx = (np.arange(n_samp) % 256).reshape(-1, 1)
    accel = np.zeros((n_samp, 3))
    arr = np.hstack([sample_idx, eeg_uv, accel])
    header = (
        "%OpenBCI Raw EXG Data\n"
        f"%Number of channels = {n_ch}\n"
        f"%Sample Rate = {int(sfreq)} Hz\n"
        "%Board = Cyton + Daisy\n"
        "%Notes = MES protocol test (60s rest + MI trials)\n"
    )
    df = pd.DataFrame(arr)
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        df.to_csv(f, index=False, header=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/protocol_test_openbci.txt"),
    )
    p.add_argument("--trials", type=int, default=12)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = build_protocol_recording(n_trials=args.trials)
    write_openbci_txt(args.output, data)
    dur = data.shape[1] / SFREQ
    print(f"Wrote {args.output} ({dur:.1f} s, {data.shape[1]} samples)")


if __name__ == "__main__":
    main()
