"""Score a recording or epoch array with production ONNX + fitted MES weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from mes_core.artifacts import load_mes_weights, load_population_baseline
from mes_core.models.inference import resolve_session_posterior
from mes_core.preprocessing import PreprocessConfig, epoch_raw, preprocess_raw
from mes_core.quality import assess_session, reliability_tier
from mes_core.scoring import MesScoreResult, compute_mes, fit_subject_baseline
from mes_core.scoring.rest import split_rest_and_task_epochs

log = structlog.get_logger(__name__)


@dataclass
class SessionScoreResult:
    """Full scoring output for one session."""

    mes: MesScoreResult
    model_sha: str
    baseline_kind: str
    n_rest_epochs: int
    n_task_epochs: int
    quality: dict[str, Any] = field(default_factory=dict)
    reliability: str = "Medium"
    cohort: str = "healthy"
    posterior_entropy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out = self.mes.to_dict()
        out["model_sha"] = self.model_sha
        out["baseline_kind"] = self.baseline_kind
        out["n_rest_epochs"] = self.n_rest_epochs
        out["n_task_epochs"] = self.n_task_epochs
        out["quality"] = self.quality
        out["reliability"] = self.reliability
        out["cohort"] = self.cohort
        out["posterior_entropy"] = self.posterior_entropy
        return out


def score_epochs(
    epochs_data: np.ndarray,
    *,
    sfreq: float,
    ch_names: list[str],
    task: str,
    rest_mask: np.ndarray | None = None,
    use_onnx: bool = True,
    cohort: str = "healthy",
    require_quality: bool = True,
) -> SessionScoreResult:
    """Score preprocessed epoch tensor (n_epochs, n_ch, n_times)."""
    data = np.asarray(epochs_data, dtype=float)
    sq, usable_mask = assess_session(data, ch_names, sfreq=sfreq)
    if require_quality and not sq.ok:
        raise ValueError(
            "Session quality too low for scoring: " + ", ".join(sq.reasons or ["unknown"])
        )

    if rest_mask is None:
        rest_idx, task_idx, kind = split_rest_and_task_epochs(data, sfreq=sfreq)
    else:
        rest_mask = np.asarray(rest_mask, dtype=bool)
        rest_idx = np.where(rest_mask)[0]
        task_idx = np.where(~rest_mask)[0]
        kind = "explicit_mask"

    rest_idx = np.asarray(rest_idx, dtype=int)
    task_idx = np.asarray(task_idx, dtype=int)
    if len(task_idx):
        task_idx = task_idx[usable_mask[task_idx]]
    if len(rest_idx):
        rest_idx = rest_idx[usable_mask[rest_idx]]

    if len(task_idx) == 0:
        task_idx = np.where(usable_mask)[0]
        if len(task_idx) == 0:
            raise ValueError("No usable epochs after quality gate")
        rest_idx = np.array([], dtype=int)
        kind = "per_trial"

    task_data = data[task_idx]
    rest_data = data[rest_idx] if len(rest_idx) else None

    if rest_data is not None and len(rest_data) >= 2:
        baseline = fit_subject_baseline(rest_data, sfreq, ch_names, task)
        baseline_kind = "subject_rest"
    elif kind == "per_trial":
        baseline = load_population_baseline(task)
        baseline_kind = "per_trial"
    else:
        baseline = load_population_baseline(task)
        baseline_kind = "population"

    p_entropy: float | None = None
    if use_onnx:
        try:
            p_model, model_sha = resolve_session_posterior(task_data, task)
            p_clip = np.clip(p_model, 1e-6, 1 - 1e-6)
            p_entropy = float(-np.mean(
                p_clip * np.log(p_clip) + (1 - p_clip) * np.log(1 - p_clip)
            ))
        except Exception as e:
            log.warning("onnx_unavailable_heuristic", err=str(e))
            p_model = _heuristic_posterior(task_data, ch_names, sfreq)
            model_sha = "heuristic"
    else:
        p_model = np.full(len(task_data), 0.5)
        model_sha = "no_model"

    weights = load_mes_weights(task, cohort=cohort)
    mes = compute_mes(
        epochs_data=task_data,
        sfreq=sfreq,
        ch_names=ch_names,
        task=task,
        baseline=baseline,
        weights=weights,
        p_model=p_model,
        rest_epochs_data=rest_data,
    )
    rel = reliability_tier(sq, baseline_kind=baseline_kind, posterior_entropy=p_entropy)
    return SessionScoreResult(
        mes=mes,
        model_sha=model_sha,
        baseline_kind=baseline_kind,
        n_rest_epochs=len(rest_idx),
        n_task_epochs=len(task_idx),
        quality=sq.to_dict(),
        reliability=rel,
        cohort=cohort,
        posterior_entropy=p_entropy,
    )


def score_recording(
    path: str | Path,
    *,
    task: str = "right_hand",
    preprocess_cfg: PreprocessConfig | None = None,
    use_onnx: bool = True,
    cohort: str = "healthy",
    require_quality: bool = True,
) -> SessionScoreResult:
    """Load, preprocess, epoch, and score one EEG file."""
    from mes_core.io import load_eeg

    raw = load_eeg(str(path))
    n_ch = len(raw.info["ch_names"])
    cfg = preprocess_cfg or PreprocessConfig(do_ica=n_ch >= 32)
    raw_pp = preprocess_raw(raw, cfg)
    epochs = epoch_raw(raw_pp)
    data = epochs.get_data() if epochs is not None and len(epochs) > 0 else None
    if data is None or len(data) == 0:
        arr = raw_pp.get_data()
        n_ch, n_t = arr.shape
        n_win = int(6.0 * raw_pp.info["sfreq"])
        if n_t < n_win:
            pad = np.zeros((n_ch, n_win - n_t), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=1)
        data = arr[None, ...]

    return score_epochs(
        data,
        sfreq=float(raw_pp.info["sfreq"]),
        ch_names=list(raw_pp.info["ch_names"]),
        task=task,
        use_onnx=use_onnx,
        cohort=cohort,
        require_quality=require_quality,
    )


def _heuristic_posterior(
    epoch_data: np.ndarray, ch_names: list[str], sfreq: float
) -> np.ndarray:
    from mes_core.config import BANDS
    from mes_core.features.bandpower import erd_percent
    from mes_core.features.lateralization import default_contra_ipsi_for_task

    contra_chs, _ = default_contra_ipsi_for_task("right_hand")
    contra_idx = [ch_names.index(c) for c in contra_chs if c in ch_names]
    if not contra_idx:
        return np.full(epoch_data.shape[0], 0.5)
    half = epoch_data.shape[-1] // 2
    baseline = epoch_data[..., :half]
    task_seg = epoch_data[..., half:]
    erd = erd_percent(task_seg, baseline, sfreq, BANDS["mu"])
    contra_erd = erd[..., contra_idx].mean(axis=-1)
    return 1.0 / (1.0 + np.exp(-np.clip((contra_erd - 30) / 20.0, -20, 20)))
