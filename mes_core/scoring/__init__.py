"""MES scoring: compute the Motor Engagement Signal from features + model output."""

from mes_core.scoring.mes import (
    MesScoreResult,
    MesWeights,
    SubjectBaseline,
    compute_mes,
    fit_mes_weights,
    fit_subject_baseline,
)

__all__ = [
    "MesScoreResult",
    "MesWeights",
    "SubjectBaseline",
    "compute_mes",
    "fit_mes_weights",
    "fit_subject_baseline",
]
