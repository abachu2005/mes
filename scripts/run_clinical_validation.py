#!/usr/bin/env python3
"""Fit stroke weights, run clinical validation, write reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed_stroke"))
    ap.add_argument("--preprocess", action="store_true", help="Run Liu2024 preprocess first")
    ap.add_argument("--max-subjects", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("validation_out/clinical"))
    ap.add_argument("--clinical-tsv", type=Path, default=None)
    ap.add_argument("--fetch-clinical", action="store_true")
    ap.add_argument("--skip-fit", action="store_true")
    args = ap.parse_args()

    if args.preprocess:
        from scripts.preprocess_moabb_datasets import preprocess_stroke

        args.data_dir.mkdir(parents=True, exist_ok=True)
        preprocess_stroke(args.data_dir, max_subjects=args.max_subjects)

    if args.fetch_clinical:
        import subprocess

        subprocess.run(
            [sys.executable, str(ROOT / "scripts/fetch_liu2024_clinical.py")],
            check=False,
        )

    if not args.skip_fit and list(args.data_dir.glob("liu2024_*.parquet")):
        from scripts.fit_mes_weights import fit_from_trials

        try:
            fit_from_trials(args.data_dir, max_files=None, cohort="stroke")
        except SystemExit as e:
            print("Stroke weight fit warning:", e)

    from mes_core.eval.clinical import (
        CohortClinicalSummary,
        ClinicalValidationReport,
        _DISCLAIMER,
        _score_trials_stroke,
        load_liu2024_clinical_table,
        run_validation,
        write_clinical_report,
    )
    from mes_core.eval.outcomes import correlate_mes_with_outcomes
    from mes_core.eval.parquet import download_processed_cache, load_parquet_dir
    from mes_core.artifacts import load_mes_weights
    import numpy as np

    stroke_dir = args.data_dir
    if not list(stroke_dir.glob("liu2024_*.parquet")):
        print("No liu2024_*.parquet in", stroke_dir, file=sys.stderr)
        return 1

    healthy_cache = download_processed_cache(max_files=40)
    healthy_val = run_validation(
        healthy_cache, prefix="physionet_", max_files=40, cohort="healthy"
    )
    stroke_val = run_validation(stroke_dir, prefix="liu2024_", max_files=None, cohort="stroke")
    stroke_trials = load_parquet_dir(
        stroke_dir,
        prefix="liu2024_",
        labels={"right_hand", "left_hand", "rest", "break"},
    )
    clinical = load_liu2024_clinical_table(args.clinical_tsv)
    weights = load_mes_weights("right_hand", cohort="stroke")
    mes_by_subj, rpi_by_subj, rnotes = _score_trials_stroke(
        stroke_trials, weights=weights, clinical_by_subject=clinical
    )

    outcome_corr: dict = {}
    if clinical:
        for field in ("nihss", "mbi", "mrs"):
            mapped = {
                code: {field: vals.get(field)}
                for code, vals in clinical.items()
                if vals.get(field) is not None
            }
            if mapped:
                outcome_corr[field] = correlate_mes_with_outcomes(
                    rpi_by_subj, mapped, outcome_key=field
                )

    report = ClinicalValidationReport(
        healthy=CohortClinicalSummary(
            cohort="healthy",
            validation=healthy_val.to_dict(),
            per_subject_mes_mean={},
            rehab_proxy_mean=None,
            notes=["PhysioNet reference cohort (HF cache)."],
        ),
        stroke=CohortClinicalSummary(
            cohort="stroke",
            validation=stroke_val.to_dict(),
            per_subject_mes_mean=mes_by_subj,
            rehab_proxy_mean=float(np.mean(list(rpi_by_subj.values())))
            if rpi_by_subj
            else None,
            outcome_correlations=outcome_corr,
            notes=rnotes
            + [f"Liu2024 parquet subjects: {len(mes_by_subj)}."],
        ),
        disclaimer=_DISCLAIMER,
    )

    write_clinical_report(report, args.out)
    print("Wrote", args.out / "clinical_validation.json")
    if report.stroke:
        for m in report.stroke.validation.get("models", []):
            if m["name"] == "MES_0_100":
                print(f"Stroke MES AUC={m['auc']:.3f} Cohen d={report.stroke.validation.get('mes_cohen_d_task_vs_rest'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
