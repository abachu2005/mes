"""Stroke rehabilitation proxy: paretic-hand MES + clinical capacity weighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def paretic_hand_for_side(paralysis_side: str) -> str:
    """Return MI label for the paretic limb ('left_hand' | 'right_hand')."""
    s = paralysis_side.strip().lower()
    if s in ("left", "l", "left hemiplegia", "left_hemiplegia"):
        return "left_hand"
    if s in ("right", "r", "right hemiplegia", "right_hemiplegia"):
        return "right_hand"
    raise ValueError(f"Unknown paralysis side: {paralysis_side!r}")


def affected_motor_channels(paralysis_side: str) -> tuple[list[str], list[str]]:
    """(contra, ipsi) for MI of the paretic hand — lesioned hemisphere emphasis."""
    hand = paretic_hand_for_side(paralysis_side)
    from mes_core.features.lateralization import default_contra_ipsi_for_task

    return default_contra_ipsi_for_task(hand)


@dataclass
class RehabProxyResult:
    """Stroke-oriented engagement summary."""

    mes_paretic_mean: float
    mes_nonparetic_mean: float | None
    rehab_proxy_index: float
    capacity_weight: float
    n_paretic_trials: int
    n_nonparetic_trials: int
    paretic_side: str | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mes_paretic_mean": self.mes_paretic_mean,
            "mes_nonparetic_mean": self.mes_nonparetic_mean,
            "rehab_proxy_index": self.rehab_proxy_index,
            "capacity_weight": self.capacity_weight,
            "n_paretic_trials": self.n_paretic_trials,
            "n_nonparetic_trials": self.n_nonparetic_trials,
            "paretic_side": self.paretic_side,
            "notes": self.notes,
        }


def compute_rehab_proxy(
    mes_per_trial: np.ndarray,
    trial_labels: list[str],
    *,
    paralysis_side: str | None = None,
    nihss: float | None = None,
    mbi: float | None = None,
    mes_global_mean: float | None = None,
) -> RehabProxyResult:
    """Combine calibrated MES with clinical capacity for a rehab-oriented index.

    RPI = mes_paretic_mean * capacity_weight

    capacity_weight defaults to 1.0; with clinical scores:
      weight = (MBI/100) * (1 - NIHSS/42), clipped to [0.15, 1.0].
    """
    mes = np.asarray(mes_per_trial, dtype=float)
    labels = [str(x).lower() for x in trial_labels]
    notes: list[str] = []

    paretic_label: str | None = None
    if paralysis_side:
        try:
            paretic_label = paretic_hand_for_side(paralysis_side)
        except ValueError:
            notes.append(f"invalid_paralysis_side:{paralysis_side}")

    if paretic_label:
        p_mask = np.array([lab == paretic_label for lab in labels])
        np_mask = np.array(
            [lab in ("left_hand", "right_hand") and lab != paretic_label for lab in labels]
        )
    else:
        # Unknown side: use all MI trials (right + left) as engagement pool.
        p_mask = np.array([lab in ("left_hand", "right_hand") for lab in labels])
        np_mask = np.zeros(len(labels), dtype=bool)
        notes.append("paretic_side_unknown_used_all_mi_trials")

    if not p_mask.any():
        fallback = float(mes_global_mean if mes_global_mean is not None else np.mean(mes))
        return RehabProxyResult(
            mes_paretic_mean=fallback,
            mes_nonparetic_mean=None,
            rehab_proxy_index=fallback,
            capacity_weight=1.0,
            n_paretic_trials=0,
            n_nonparetic_trials=0,
            paretic_side=paralysis_side,
            notes=notes + ["no_paretic_trials_fallback_global_mes"],
        )

    mes_p = float(np.mean(mes[p_mask]))
    mes_np = float(np.mean(mes[np_mask])) if np_mask.any() else None

    cap = 1.0
    if mbi is not None and np.isfinite(mbi):
        cap *= float(np.clip(mbi / 100.0, 0.2, 1.0))
    if nihss is not None and np.isfinite(nihss):
        cap *= float(np.clip(1.0 - nihss / 42.0, 0.15, 1.0))
    cap = float(np.clip(cap, 0.15, 1.0))

    rpi = mes_p * cap
    if mbi is None and nihss is None:
        notes.append("clinical_scores_absent_rpi_equals_mes_paretic")

    return RehabProxyResult(
        mes_paretic_mean=mes_p,
        mes_nonparetic_mean=mes_np,
        rehab_proxy_index=float(rpi),
        capacity_weight=cap,
        n_paretic_trials=int(p_mask.sum()),
        n_nonparetic_trials=int(np_mask.sum()),
        paretic_side=paralysis_side,
        notes=notes,
    )
