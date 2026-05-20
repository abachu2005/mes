"""Motor Engagement Signal (MES) scoring.

Per epoch, given target task T:

    z_mu   = z(ERD_mu_contra,    subject_baseline)
    z_beta = z(ERD_beta_contra,  subject_baseline)
    z_li   = z(LateralizationIndex, subject_baseline)
    z_mrcp = z(MRCP_amplitude,   subject_baseline)
    p_model = calibrated_posterior(target=T)

    raw = w1*z_mu + w2*z_beta + w3*z_li + w4*z_mrcp + w5*logit(p_model)
    MES = 100 * sigmoid(raw)        # bounded [0, 100]

Weights w_i are fit by logistic regression of `raw_features` against
dataset-given task-vs-rest labels (task=1, rest=0), so MES weight fitting
is not circular with p_model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from mes_core.config import BANDS
from mes_core.features.bandpower import erd_percent
from mes_core.features.lateralization import default_contra_ipsi_for_task, lateralization_index
from mes_core.features.mrcp import mrcp_features


# ---------------------------------------------------------------------------
# Subject baseline (rest block stats)
# ---------------------------------------------------------------------------

@dataclass
class SubjectBaseline:
    """Per-subject (or population fallback) mean/std for each MES feature."""

    feature_names: tuple[str, ...]
    mean: np.ndarray   # shape (n_features,)
    std: np.ndarray    # shape (n_features,)

    @classmethod
    def zeros(cls, n_features: int = 4) -> "SubjectBaseline":
        names = ("z_mu", "z_beta", "z_li", "z_mrcp")[:n_features]
        return cls(feature_names=names, mean=np.zeros(n_features), std=np.ones(n_features))

    def zscore(self, x: np.ndarray) -> np.ndarray:
        m = self.mean.reshape((1,) * (x.ndim - 1) + (-1,))
        s = self.std.reshape((1,) * (x.ndim - 1) + (-1,))
        return (x - m) / np.maximum(s, 1e-9)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubjectBaseline":
        return cls(
            feature_names=tuple(d["feature_names"]),
            mean=np.asarray(d["mean"], dtype=float),
            std=np.asarray(d["std"], dtype=float),
        )


def fit_subject_baseline(
    rest_epochs_data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    task: str,
) -> SubjectBaseline:
    """Compute mean/std of the 4 MES sub-features over a rest block.

    `rest_epochs_data` is (n_rest_epochs, n_channels, n_times).
    """
    feats = _per_trial_subfeatures(
        epochs_data=rest_epochs_data,
        sfreq=sfreq,
        ch_names=ch_names,
        task=task,
        baseline_data=rest_epochs_data,
    )
    arr = np.stack([feats["z_mu"], feats["z_beta"], feats["z_li"], feats["z_mrcp"]], axis=-1)
    return SubjectBaseline(
        feature_names=("z_mu", "z_beta", "z_li", "z_mrcp"),
        mean=arr.mean(axis=0),
        std=arr.std(axis=0) + 1e-6,
    )


# ---------------------------------------------------------------------------
# MES weights
# ---------------------------------------------------------------------------

@dataclass
class MesWeights:
    """Linear weights for the MES combination, plus an intercept."""

    w_mu: float = 1.0
    w_beta: float = 1.0
    w_li: float = 1.0
    w_mrcp: float = 0.5
    w_pmodel: float = 2.0
    intercept: float = 0.0
    feature_names: tuple[str, ...] = field(
        default=("z_mu", "z_beta", "z_li", "z_mrcp", "logit_pmodel")
    )

    @classmethod
    def default(cls) -> "MesWeights":
        return cls()

    def as_vector(self) -> np.ndarray:
        return np.array([self.w_mu, self.w_beta, self.w_li, self.w_mrcp, self.w_pmodel])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MesWeights":
        d = dict(d)
        d.pop("feature_names", None)
        return cls(**d)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _per_trial_subfeatures(
    epochs_data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    task: str,
    baseline_data: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute the 4 raw sub-features (un-z-scored) per epoch."""
    data = np.asarray(epochs_data, dtype=float)
    n_epochs, _n_ch, n_times = data.shape

    # Split each epoch into baseline (first half-second) and task (last second)
    # if no separate baseline_data provided.
    half = max(1, int(0.5 * sfreq))
    last = max(1, int(1.0 * sfreq))
    if baseline_data is None:
        baseline = data[..., :half]
    else:
        baseline = baseline_data[..., -half:]
        # Ensure broadcasting works even when shape (n_rest,...) != (n_trials,...)
        if baseline.shape[0] != n_epochs:
            baseline = np.broadcast_to(
                baseline.mean(axis=0, keepdims=True), (n_epochs, *baseline.shape[1:])
            )
    task_seg = data[..., -last:]

    contra, ipsi = default_contra_ipsi_for_task(task)

    erd_mu = erd_percent(task_seg, baseline, sfreq, BANDS["mu"])
    erd_beta = erd_percent(task_seg, baseline, sfreq, BANDS["beta"])

    contra_idx = [ch_names.index(c) for c in contra if c in ch_names]
    if contra_idx:
        mu_contra = erd_mu[..., contra_idx].mean(axis=-1)
        beta_contra = erd_beta[..., contra_idx].mean(axis=-1)
    else:
        mu_contra = erd_mu.mean(axis=-1)
        beta_contra = erd_beta.mean(axis=-1)

    li = lateralization_index(erd_mu, ch_names, contra_channels=contra, ipsi_channels=ipsi)
    if np.isscalar(li):
        li = np.full(n_epochs, float(li))

    mrcp = mrcp_features(data, sfreq, tmin=-(n_times / sfreq) / 2.0)
    mrcp_amp = mrcp["amplitude"]
    if mrcp_amp.ndim == 2 and contra_idx:
        mrcp_contra = mrcp_amp[..., contra_idx].mean(axis=-1)
    elif mrcp_amp.ndim == 2:
        mrcp_contra = mrcp_amp.mean(axis=-1)
    else:
        mrcp_contra = mrcp_amp

    return {
        "z_mu": np.asarray(mu_contra, dtype=float),
        "z_beta": np.asarray(beta_contra, dtype=float),
        "z_li": np.asarray(li, dtype=float),
        "z_mrcp": np.asarray(mrcp_contra, dtype=float),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class MesScoreResult:
    """Result of MES scoring for a session."""

    mes_per_trial: np.ndarray            # shape (n_trials,) in [0, 100]
    raw_features: dict[str, np.ndarray]  # un-z-scored sub-features per trial
    z_features: np.ndarray               # shape (n_trials, 4) after baseline z-score
    p_model: np.ndarray                  # shape (n_trials,) classifier posterior
    raw_combination: np.ndarray          # shape (n_trials,) before sigmoid
    summary: dict[str, float]            # session-level rollups

    def to_dict(self) -> dict[str, Any]:
        return {
            "mes_per_trial": self.mes_per_trial.tolist(),
            "raw_features": {k: v.tolist() for k, v in self.raw_features.items()},
            "z_features": self.z_features.tolist(),
            "p_model": self.p_model.tolist(),
            "raw_combination": self.raw_combination.tolist(),
            "summary": self.summary,
        }


def compute_mes(
    epochs_data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    task: str,
    baseline: SubjectBaseline,
    weights: MesWeights,
    p_model: np.ndarray,
    *,
    rest_epochs_data: np.ndarray | None = None,
) -> MesScoreResult:
    """Compute the Motor Engagement Signal for each trial in a session.

    Parameters
    ----------
    epochs_data : (n_epochs, n_channels, n_times) preprocessed epochs in volts.
    sfreq : sampling frequency in Hz.
    ch_names : channel name list (used to identify contra/ipsi).
    task : task name ('right_hand', 'left_hand', 'feet', 'gait', etc.).
    baseline : per-subject baseline statistics. Use `SubjectBaseline.zeros(4)` to skip.
    weights : MES combination weights.
    p_model : per-trial classifier posterior probability for the target class.
    rest_epochs_data : optional rest epochs used for ERD baseline (else uses each
        trial's own pre-task segment).
    """
    epochs_data = np.asarray(epochs_data, dtype=float)
    p_model = np.asarray(p_model, dtype=float)

    raw_sub = _per_trial_subfeatures(
        epochs_data=epochs_data,
        sfreq=sfreq,
        ch_names=ch_names,
        task=task,
        baseline_data=rest_epochs_data,
    )
    arr = np.stack(
        [raw_sub["z_mu"], raw_sub["z_beta"], raw_sub["z_li"], raw_sub["z_mrcp"]],
        axis=-1,
    )  # (n_epochs, 4)
    z_arr = baseline.zscore(arr)

    logit_p = _logit(p_model)
    raw_combo = (
        weights.w_mu * z_arr[:, 0]
        + weights.w_beta * z_arr[:, 1]
        + weights.w_li * z_arr[:, 2]
        + weights.w_mrcp * z_arr[:, 3]
        + weights.w_pmodel * logit_p
        + weights.intercept
    )
    mes = 100.0 * _sigmoid(raw_combo)

    summary: dict[str, float] = {
        "mes_mean": float(np.mean(mes)),
        "mes_median": float(np.median(mes)),
        "mes_std": float(np.std(mes)),
        "n_trials": int(mes.shape[0]),
    }
    return MesScoreResult(
        mes_per_trial=mes,
        raw_features=raw_sub,
        z_features=z_arr,
        p_model=p_model,
        raw_combination=raw_combo,
        summary=summary,
    )


def fit_mes_weights(
    z_features: np.ndarray,
    p_model: np.ndarray,
    labels: np.ndarray,
    *,
    C: float = 1.0,
) -> MesWeights:
    """Fit MES weights via logistic regression of `[z, logit(p)]` vs labels.

    `labels` are 1 for task epochs, 0 for rest epochs (dataset-given,
    not derived from the classifier).
    """
    from sklearn.linear_model import LogisticRegression

    z_features = np.asarray(z_features, dtype=float)
    p_model = np.asarray(p_model, dtype=float)
    labels = np.asarray(labels).astype(int).ravel()

    X = np.concatenate([z_features, _logit(p_model)[:, None]], axis=1)
    clf = LogisticRegression(C=C, max_iter=1000)
    clf.fit(X, labels)
    coefs = clf.coef_.ravel()
    return MesWeights(
        w_mu=float(coefs[0]),
        w_beta=float(coefs[1]),
        w_li=float(coefs[2]),
        w_mrcp=float(coefs[3]),
        w_pmodel=float(coefs[4]),
        intercept=float(clf.intercept_[0]),
    )
