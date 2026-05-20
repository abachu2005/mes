"""MES scoring: compute the Motor Engagement Signal from features + model output."""

from mes_core.scoring.mes import (
    MesWeights,
    MesScoreResult,
    compute_mes,
    fit_mes_weights,
    SubjectBaseline,
    fit_subject_baseline,
)

__all__ = [
    "MesWeights",
    "MesScoreResult",
    "compute_mes",
    "fit_mes_weights",
    "SubjectBaseline",
    "fit_subject_baseline",
]
