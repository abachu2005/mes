"""Validation harness: MES vs classifier-only baselines on processed parquet."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from mes_core.artifacts import load_mes_weights
from mes_core.config import OPENBCI_MONTAGE_16, TARGET_SFREQ
from mes_core.eval.metrics import brier_score
from mes_core.eval.parquet import ParquetTrial, load_parquet_dir
from mes_core.models.inference import resolve_session_posterior
from mes_core.scoring import SubjectBaseline, compute_mes, fit_mes_weights, fit_subject_baseline


@dataclass
class ModelEvalRow:
    name: str
    accuracy: float
    auc: float
    brier: float
    n: int


@dataclass
class ValidationReport:
    """Structured output from ``run_validation``."""

    n_trials: int
    n_subjects: int
    models: list[ModelEvalRow]
    mes_cohen_d_task_vs_rest: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trials": self.n_trials,
            "n_subjects": self.n_subjects,
            "models": [asdict(m) for m in self.models],
            "mes_cohen_d_task_vs_rest": self.mes_cohen_d_task_vs_rest,
            "notes": self.notes,
        }


def _trial_features_and_posterior(
    trials: list[ParquetTrial],
    *,
    population_baseline: SubjectBaseline,
    cohort: str = "healthy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return z_features (n,4), p_model, labels, mes_scores."""
    ch = list(OPENBCI_MONTAGE_16)
    sfreq = TARGET_SFREQ
    weights = load_mes_weights("right_hand", cohort=cohort)

    z_list: list[np.ndarray] = []
    p_list: list[float] = []
    y_list: list[int] = []
    mes_list: list[float] = []

    # Group rest epochs per subject for baseline
    by_subj: dict[str, list[np.ndarray]] = {}
    for t in trials:
        if t.label in ("rest", "break"):
            by_subj.setdefault(t.subject, []).append(t.X[None, ...])

    for t in trials:
        if t.label not in ("right_hand", "rest", "break"):
            continue
        rest_stack = None
        if by_subj.get(t.subject):
            rest_stack = np.concatenate(by_subj[t.subject], axis=0)
        bl = (
            fit_subject_baseline(rest_stack, sfreq, ch, "right_hand")
            if rest_stack is not None and len(rest_stack) >= 2
            else population_baseline
        )
        x = t.X[None, ...]
        try:
            p, _ = resolve_session_posterior(x, "right_hand", cohort=cohort)
            p_val = float(p[0])
        except Exception:
            p_val = 0.5
        res = compute_mes(
            x, sfreq, ch, "right_hand", bl, weights, np.array([p_val]),
            rest_epochs_data=rest_stack,
        )
        z_list.append(res.z_features[0])
        p_list.append(p_val)
        y_list.append(t.y)
        mes_list.append(float(res.mes_per_trial[0]))

    return (
        np.stack(z_list),
        np.asarray(p_list),
        np.asarray(y_list),
        np.asarray(mes_list),
    )


def run_validation(
    data_dir: Path,
    *,
    prefix: str = "physionet_",
    max_files: int | None = 30,
    cohort: str = "healthy",
) -> ValidationReport:
    """Compare posteriors, MES, and accuracy-only baselines on labeled parquet."""
    from mes_core.artifacts import load_population_baseline

    trials = load_parquet_dir(
        data_dir,
        prefix=prefix,
        labels={"right_hand", "rest", "break"},
        max_files=max_files,
    )
    if len(trials) < 20:
        raise RuntimeError(f"Need >=20 trials, got {len(trials)}")

    pop_bl = load_population_baseline("right_hand", cohort=cohort)
    z, p_model, labels, mes_scores = _trial_features_and_posterior(
        trials, population_baseline=pop_bl, cohort=cohort
    )

    notes: list[str] = []
    models: list[ModelEvalRow] = []

    def _metrics(scores: np.ndarray, name: str) -> ModelEvalRow:
        if (labels == 1).any():
            task_med = float(np.median(scores[labels == 1]))
            pred = (scores >= task_med).astype(int)
        else:
            pred = labels.copy()
        try:
            auc = float(roc_auc_score(labels, scores))
        except ValueError:
            auc = float("nan")
        probs = scores / 100.0 if scores.max() > 1.5 else scores
        return ModelEvalRow(
            name=name,
            accuracy=float(accuracy_score(labels, pred)),
            auc=auc,
            brier=brier_score(probs, labels),
            n=len(labels),
        )

    models.append(_metrics(p_model, "ensemble_posterior"))
    models.append(_metrics(mes_scores, "MES_0_100"))

    # MES feature-only (no p_model weight contribution) via refit on z only
    w0 = fit_mes_weights(z, np.full(len(labels), 0.5), labels)
    raw = (
        w0.w_mu * z[:, 0]
        + w0.w_beta * z[:, 1]
        + w0.w_li * z[:, 2]
        + w0.w_mrcp * z[:, 3]
        + w0.intercept
    )
    feat_only = 100.0 / (1.0 + np.exp(-raw))
    models.append(_metrics(feat_only, "MES_features_only"))

    task_mes = mes_scores[labels == 1]
    rest_mes = mes_scores[labels == 0]
    if len(task_mes) and len(rest_mes):
        pooled = np.sqrt(
            (task_mes.var(ddof=1) + rest_mes.var(ddof=1)) / 2.0
        )
        d = float((task_mes.mean() - rest_mes.mean()) / pooled) if pooled > 0 else 0.0
    else:
        d = 0.0
    notes.append("Labels from dataset parquet (right_hand=1, rest=0).")
    notes.append("ensemble_posterior uses production ONNX mean posterior.")

    subjects = {t.subject for t in trials}
    return ValidationReport(
        n_trials=len(trials),
        n_subjects=len(subjects),
        models=models,
        mes_cohen_d_task_vs_rest=float(d),
        notes=notes,
    )


def write_validation_report(report: ValidationReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    md = ["# MES validation report\n", f"Trials: {report.n_trials}, subjects: {report.n_subjects}\n\n"]
    md.append("| Model | n | Accuracy | AUC | Brier |\n|---|---:|---:|---:|---:|\n")
    for m in report.models:
        md.append(f"| {m.name} | {m.n} | {m.accuracy:.3f} | {m.auc:.3f} | {m.brier:.3f} |\n")
    md.append(f"\nMES Cohen's d (task vs rest): **{report.mes_cohen_d_task_vs_rest:.3f}**\n")
    for note in report.notes:
        md.append(f"\n- {note}")
    (out_dir / "validation_report.md").write_text("".join(md), encoding="utf-8")
