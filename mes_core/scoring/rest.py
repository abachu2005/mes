"""Heuristics to separate rest vs task epochs when labels are absent."""

from __future__ import annotations

import numpy as np


def split_rest_and_task_epochs(
    epochs_data: np.ndarray,
    *,
    sfreq: float,
    max_rest_fraction: float = 0.25,
    min_rest: int = 1,
    max_rest: int = 8,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split epoch indices into rest (baseline) vs task (scored) sets.

    Strategy (in order):
    1. If the first ``max_rest`` epochs have substantially lower band power variance
       than later epochs, treat them as an implicit rest block.
    2. Otherwise score all epochs and use per-trial baseline inside ``compute_mes``.
    """
    data = np.asarray(epochs_data, dtype=float)
    n = data.shape[0]
    if n <= 2:
        return np.array([], dtype=int), np.arange(n), "per_trial"

    n_rest = min(max_rest, max(min_rest, int(n * max_rest_fraction)))
    early = data[:n_rest]
    late = data[n_rest:]
    early_var = float(np.var(early))
    late_var = float(np.var(late))
    # Rest blocks are often calmer (lower variance) at session start.
    if n_rest >= min_rest and early_var < 0.85 * late_var:
        rest_idx = np.arange(n_rest)
        task_idx = np.arange(n_rest, n)
        return rest_idx, task_idx, "implicit_rest_block"

    return np.array([], dtype=int), np.arange(n), "per_trial"


def rest_mask_protocol(
    n_epochs: int,
    *,
    sfreq: float,
    rest_seconds: float = 60.0,
    window_s: float = 6.0,
    step_s: float = 3.0,
) -> np.ndarray:
    """Boolean mask: True for sliding-window epochs in the protocol rest block.

    Matches ``epoch_sliding_windows``: epoch *i* starts at ``i * step_s`` seconds.
    Windows that start before ``rest_seconds`` are treated as rest (baseline).
    """
    del sfreq  # reserved for sample-accurate alignment later
    n = int(n_epochs)
    if n <= 0:
        return np.zeros(0, dtype=bool)
    starts = np.arange(n, dtype=float) * float(step_s)
    return starts < float(rest_seconds)
