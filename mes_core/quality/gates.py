"""Epoch- and session-level quality gates for trustworthy MES."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class EpochQuality:
    index: int
    ok: bool
    score: float  # 0-1 higher is better
    reasons: list[str]


@dataclass
class SessionQuality:
    n_epochs: int
    n_usable: int
    fraction_usable: float
    mean_epoch_score: float
    flatline_channels: list[str]
    excessive_variance: bool
    ok: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_epoch(
    epoch: np.ndarray,
    ch_names: list[str],
    *,
    sfreq: float,
) -> EpochQuality:
    """Score one epoch (n_ch, n_times)."""
    x = np.asarray(epoch, dtype=float)
    reasons: list[str] = []
    score = 1.0

    peak = float(np.max(np.abs(x)))
    if peak < 1e-7:
        return EpochQuality(0, False, 0.0, ["flatline"])
    if peak > 500e-6:
        reasons.append("clipping_suspected")
        score -= 0.4

    var_ch = x.var(axis=-1)
    dead = [ch_names[i] for i in range(len(ch_names)) if var_ch[i] < 1e-14]
    if len(dead) > len(ch_names) // 3:
        reasons.append("many_dead_channels")
        score -= 0.5

    # Line noise proxy: narrowband 50/60 Hz energy ratio
    if x.shape[-1] > int(2 * sfreq):
        from scipy.signal import welch

        f, pxx = welch(x.mean(axis=0), fs=sfreq, nperseg=min(256, x.shape[-1]))
        line = ((pxx[(f > 49) & (f < 51)].sum()) + (pxx[(f > 59) & (f < 61)].sum()))
        total = pxx.sum() + 1e-12
        if line / total > 0.35:
            reasons.append("line_noise")
            score -= 0.25

    score = max(0.0, min(1.0, score))
    ok = score >= 0.45 and "flatline" not in reasons
    return EpochQuality(0, ok, score, reasons)


def assess_session(
    epochs_data: np.ndarray,
    ch_names: list[str],
    *,
    sfreq: float,
    min_usable_fraction: float = 0.5,
) -> tuple[SessionQuality, np.ndarray]:
    """Return session quality and boolean mask of usable epochs."""
    data = np.asarray(epochs_data, dtype=float)
    n = data.shape[0]
    per_epoch: list[EpochQuality] = []
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        eq = assess_epoch(data[i], ch_names, sfreq=sfreq)
        eq.index = i
        per_epoch.append(eq)
        if not eq.ok:
            mask[i] = False

    var_all = float(data.var())
    flat_ch = []
    for i, name in enumerate(ch_names):
        if data[:, i, :].var() < 1e-14:
            flat_ch.append(name)

    reasons: list[str] = []
    frac = float(mask.mean()) if n else 0.0
    mean_sc = float(np.mean([e.score for e in per_epoch])) if per_epoch else 0.0
    excessive = var_all > 1e-3 or var_all < 1e-12

    if frac < min_usable_fraction:
        reasons.append("low_usable_epochs")
    if len(flat_ch) >= 4:
        reasons.append("flat_channels")
    if excessive and var_all > 1e-3:
        reasons.append("excessive_variance")

    ok = frac >= min_usable_fraction and len(flat_ch) < 6 and "flatline" not in reasons

    sq = SessionQuality(
        n_epochs=n,
        n_usable=int(mask.sum()),
        fraction_usable=frac,
        mean_epoch_score=mean_sc,
        flatline_channels=flat_ch,
        excessive_variance=excessive,
        ok=ok,
        reasons=reasons,
    )
    return sq, mask


def reliability_tier(
    session_quality: SessionQuality,
    *,
    baseline_kind: str,
    posterior_entropy: float | None = None,
) -> str:
    """Map diagnostics to High / Medium / Low for UI."""
    if not session_quality.ok:
        return "Low"
    score = session_quality.fraction_usable * session_quality.mean_epoch_score
    if baseline_kind == "subject_rest":
        score += 0.15
    if posterior_entropy is not None and posterior_entropy > 0.9:
        score -= 0.1
    if score >= 0.75:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"
