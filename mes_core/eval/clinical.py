"""Open-dataset clinical validation (stroke cohort + rehab proxy)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mes_core.artifacts import load_mes_weights, weights_bundle_path, write_weights_bundle
from mes_core.config import OPENBCI_MONTAGE_16, TARGET_SFREQ
from mes_core.eval.outcomes import correlate_mes_with_outcomes, load_outcomes_csv
from mes_core.eval.parquet import ParquetTrial, load_parquet_dir
from mes_core.eval.validate import ModelEvalRow, run_validation, write_validation_report
from mes_core.models.inference import resolve_session_posterior
from mes_core.scoring import MesWeights, compute_mes, fit_mes_weights, fit_subject_baseline
from mes_core.scoring.rehab_proxy import compute_rehab_proxy


@dataclass
class CohortClinicalSummary:
    cohort: str
    validation: dict[str, Any]
    per_subject_mes_mean: dict[str, float]
    rehab_proxy_mean: float | None
    outcome_correlations: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class ClinicalValidationReport:
    healthy: CohortClinicalSummary | None
    stroke: CohortClinicalSummary | None
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": asdict(self.healthy) if self.healthy else None,
            "stroke": asdict(self.stroke) if self.stroke else None,
            "disclaimer": self.disclaimer,
        }


def _score_trials_stroke(
    trials: list[ParquetTrial],
    *,
    weights: MesWeights,
    clinical_by_subject: dict[str, dict[str, float]],
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Per-subject mean MES and rehab proxy index on Liu-style trials."""
    ch = list(OPENBCI_MONTAGE_16)
    sfreq = TARGET_SFREQ
    by_subj_rest: dict[str, list[np.ndarray]] = {}
    by_subj_trials: dict[str, list[ParquetTrial]] = {}
    for t in trials:
        if t.label in ("rest", "break"):
            by_subj_rest.setdefault(t.subject, []).append(t.X)
        if t.label in ("right_hand", "left_hand"):
            by_subj_trials.setdefault(t.subject, []).append(t)

    mes_means: dict[str, float] = {}
    rpi_means: dict[str, float] = {}
    notes: list[str] = []

    for subj, tlist in by_subj_trials.items():
        rest_stack = by_subj_rest.get(subj)
        if rest_stack and len(rest_stack) >= 2:
            bl = fit_subject_baseline(np.stack(rest_stack), sfreq, ch, "right_hand")
        else:
            bl = fit_subject_baseline(np.zeros((2, len(ch), 100)), sfreq, ch, "right_hand")

        xs = np.stack([t.X for t in tlist])
        labels = [t.label for t in tlist]
        try:
            p, _ = resolve_session_posterior(xs, "right_hand", cohort="stroke")
        except Exception:
            p = np.full(len(tlist), 0.5)

        base = subj.split("_")[0] if "_" in subj else subj
        clin = (
            clinical_by_subject.get(subj)
            or clinical_by_subject.get(base)
            or clinical_by_subject.get(base.lstrip("S"))
            or {}
        )
        side = clin.get("paralysis_side") or clin.get("ParalysisSide")
        if isinstance(side, str):
            side = side.strip()
        elif side is not None:
            side = str(side).strip()
        mes = compute_mes(
            xs,
            sfreq,
            ch,
            "right_hand",
            bl,
            weights,
            p,
            rest_epochs_data=(np.stack(rest_stack) if rest_stack else None),
            paralysis_side=side,
        )
        rehab = compute_rehab_proxy(
            mes.mes_per_trial,
            labels,
            paralysis_side=side,
            nihss=clin.get("nihss") or clin.get("NIHSS"),
            mbi=clin.get("mbi") or clin.get("MBI"),
            mes_global_mean=float(mes.summary["mes_mean"]),
        )
        mes_means[subj] = float(mes.summary["mes_mean"])
        rpi_means[subj] = float(rehab.rehab_proxy_index)

    if not clinical_by_subject:
        notes.append("no_per_subject_clinical_table_rpi_equals_mes_heuristic")
    return mes_means, rpi_means, notes


def load_liu2024_clinical_table(path: Path | None = None) -> dict[str, dict[str, float]]:
    """Load per-subject clinical fields keyed by subject id (e.g. S1 or 1)."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "liu2024_clinical.tsv"
    if not path.exists():
        return {}
    import pandas as pd

    df = pd.read_csv(path, sep="\t")
    out: dict[str, dict[str, float]] = {}
    id_col = "participant_id" if "participant_id" in df.columns else "Participant_ID"
    for _, row in df.iterrows():
        pid = str(row.get(id_col, "")).strip().replace("sub-", "")
        if not pid:
            continue
        if pid.isdigit():
            key = f"S{int(pid)}"
        elif pid.startswith("S"):
            key = pid
        else:
            key = f"S{pid}"
        rec: dict[str, float] = {}
        for col in df.columns:
            cl = col.lower()
            if cl in ("participant_id", "participant_code", "code"):
                continue
            val = row[col]
            if isinstance(val, (int, float)) and np.isfinite(val):
                rec[cl] = float(val)
            elif isinstance(val, str) and val.strip():
                rec[cl] = val  # type: ignore[assignment]
        out[key] = rec
    return out


def run_clinical_validation(
    data_dir: Path,
    *,
    clinical_tsv: Path | None = None,
    max_files: int | None = None,
) -> ClinicalValidationReport:
    """Validate healthy (PhysioNet) and stroke (Liu2024) cohorts + rehab proxy."""
    notes_stroke: list[str] = []
    clinical = load_liu2024_clinical_table(clinical_tsv)

    healthy_report = run_validation(
        data_dir, prefix="physionet_", max_files=max_files, cohort="healthy"
    )
    healthy_summary = CohortClinicalSummary(
        cohort="healthy",
        validation=healthy_report.to_dict(),
        per_subject_mes_mean={},
        rehab_proxy_mean=None,
        notes=["PhysioNet open MI — reference cohort."],
    )

    stroke_trials = load_parquet_dir(
        data_dir,
        prefix="liu2024_",
        labels={"right_hand", "left_hand", "rest", "break"},
        max_files=max_files,
    )
    if len(stroke_trials) < 40:
        return ClinicalValidationReport(
            healthy=healthy_summary,
            stroke=None,
            disclaimer=_DISCLAIMER,
        )

    stroke_val = run_validation(
        data_dir, prefix="liu2024_", max_files=max_files, cohort="stroke"
    )
    weights = load_mes_weights("right_hand", cohort="stroke")
    mes_by_subj, rpi_by_subj, rnotes = _score_trials_stroke(
        stroke_trials, weights=weights, clinical_by_subject=clinical
    )
    notes_stroke.extend(rnotes)

    outcome_corr: dict[str, Any] = {}
    if clinical:
        # Map clinical keys to participant codes used in correlation helper.
        clin_for_corr = {
            k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
            for k, v in clinical.items()
        }
        for key_name, field in (("nihss", "NIHSS"), ("mbi", "MBI"), ("mrs", "mRS")):
            mapped = {
                code: {field: vals.get(field.lower()) or vals.get(field)}
                for code, vals in clin_for_corr.items()
                if (vals.get(field.lower()) or vals.get(field)) is not None
            }
            if mapped:
                outcome_corr[field] = correlate_mes_with_outcomes(
                    rpi_by_subj, mapped, outcome_key=field
                )

    stroke_summary = CohortClinicalSummary(
        cohort="stroke",
        validation=stroke_val.to_dict(),
        per_subject_mes_mean=mes_by_subj,
        rehab_proxy_mean=float(np.mean(list(rpi_by_subj.values()))) if rpi_by_subj else None,
        outcome_correlations=outcome_corr,
        notes=notes_stroke
        + [
            "Stroke weights + population baseline from mes_weights_right_hand_stroke.json.",
            "Rehab Proxy Index (RPI) = paretic-hand MES × capacity (MBI, NIHSS) when clinical TSV present.",
        ],
    )

    return ClinicalValidationReport(
        healthy=healthy_summary,
        stroke=stroke_summary,
        disclaimer=_DISCLAIMER,
    )


_DISCLAIMER = (
    "Research validation on open acute-stroke MI data (Liu2024). "
    "Not FDA/CE cleared; not a substitute for prospective FMA-linked clinical trials."
)


def write_clinical_report(report: ClinicalValidationReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clinical_validation.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    lines = [
        "# Clinical validation (open datasets)\n\n",
        report.disclaimer + "\n\n",
    ]
    for label, block in (("Healthy", report.healthy), ("Stroke", report.stroke)):
        if block is None:
            continue
        lines.append(f"## {label} ({block.cohort})\n\n")
        v = block.validation
        lines.append(f"- Trials: {v.get('n_trials')}, subjects: {v.get('n_subjects')}\n")
        lines.append(f"- MES Cohen's d: {v.get('mes_cohen_d_task_vs_rest')}\n")
        if block.rehab_proxy_mean is not None:
            lines.append(f"- Mean Rehab Proxy Index: **{block.rehab_proxy_mean:.2f}**\n")
        for m in v.get("models", []):
            lines.append(
                f"- {m['name']}: AUC={m['auc']:.3f}, acc={m['accuracy']:.3f}\n"
            )
        for note in block.notes:
            lines.append(f"- {note}\n")
        if block.outcome_correlations:
            lines.append(f"- Outcome correlations: `{block.outcome_correlations}`\n")
    (out_dir / "clinical_validation.md").write_text("".join(lines), encoding="utf-8")
