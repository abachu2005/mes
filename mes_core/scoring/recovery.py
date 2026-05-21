"""Longitudinal recovery index relative to participant history."""

from __future__ import annotations

import numpy as np


def mes_recovery_z(
    current_mes_mean: float,
    prior_mes_means: list[float],
    *,
    min_prior: int = 1,
) -> tuple[float | None, str]:
    """Z-score current session MES vs participant's prior completed sessions.

    Returns (recovery_z, label). ``None`` if insufficient history.
    """
    if len(prior_mes_means) < min_prior:
        return None, "insufficient_history"

    prior = np.asarray(prior_mes_means, dtype=float)
    mu = float(prior.mean())
    sd = float(prior.std(ddof=1)) if len(prior) > 1 else 8.0
    sd = max(sd, 5.0)
    z = (current_mes_mean - mu) / sd
    if z >= 0.5:
        label = "improving_vs_baseline"
    elif z <= -0.5:
        label = "below_baseline"
    else:
        label = "stable"
    return float(z), label
