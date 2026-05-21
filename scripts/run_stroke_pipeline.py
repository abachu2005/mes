#!/usr/bin/env python3
"""Train stroke ONNX, refit MES weights, validate, and write benchmark gates."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mes_core.eval.validate import run_validation, write_validation_report
from scripts.fit_mes_weights import fit_from_trials
from scripts.gh_train_riemannian import train as train_riemannian

BUNDLED_MODELS = ROOT / "mes_core/data/models"
GATES_PATH = ROOT / "mes_core/data/stroke_validation_gates.json"
BENCHMARK_PATH = ROOT / "mes_core/data/benchmarks.json"

# Targets for Liu2024 right_hand vs break/rest (LOSO on parquet).
STROKE_MIN_AUC = 0.55
STROKE_TARGET_AUC = 0.75
STROKE_MIN_COHEN_D = 0.25
STROKE_TARGET_COHEN_D = 0.80


def _mes_row(report) -> tuple[float, float]:
    auc = 0.0
    d = report.mes_cohen_d_task_vs_rest
    for m in report.models:
        if m.name == "MES_0_100":
            auc = m.auc
    return auc, d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/processed_stroke")
    ap.add_argument("--out-models", type=Path, default=Path("/tmp/mes-stroke-models"))
    ap.add_argument("--validation-out", type=Path, default=ROOT / "validation_out/stroke")
    ap.add_argument("--min-files", type=int, default=20, help="Min liu2024 parquet files")
    ap.add_argument("--upload-models", action="store_true")
    args = ap.parse_args()

    files = sorted(args.data_dir.glob("liu2024_*.parquet"))
    if len(files) < args.min_files:
        raise SystemExit(f"Need >={args.min_files} liu2024 parquet files, got {len(files)}")

    print("=== Train stroke Riemannian ONNX ===")
    stats = train_riemannian(
        args.data_dir,
        args.out_models,
        prefix="liu2024_",
        onnx_name="riemannian_lr_right_hand_stroke.onnx",
        meta_name="riemannian_lr_right_hand_stroke.json",
        frontend_name="riemannian_lr_frontend_stroke.npz",
        cohort="stroke",
    )

    BUNDLED_MODELS.mkdir(parents=True, exist_ok=True)
    for name in (
        "riemannian_lr_right_hand_stroke.onnx",
        "riemannian_lr_right_hand_stroke.json",
        "riemannian_lr_frontend_stroke.npz",
    ):
        src = args.out_models / name
        if src.exists():
            shutil.copy2(src, BUNDLED_MODELS / name)

    print("=== Refit stroke MES weights ===")
    fit_from_trials(args.data_dir, max_files=None, cohort="stroke")

    print("=== Validate stroke cohort ===")
    report = run_validation(
        args.data_dir, prefix="liu2024_", max_files=None, cohort="stroke"
    )
    write_validation_report(report, args.validation_out)
    mes_auc, mes_d = _mes_row(report)

    bench = {
        "stroke_liu2024": {
            "n_trials": report.n_trials,
            "n_subjects": report.n_subjects,
            "mes_auc": mes_auc,
            "mes_cohen_d": mes_d,
            "models": [m.__dict__ for m in report.models],
            "riemannian_loso_acc": stats.get("loso_acc"),
        },
        "gates": {
            "min_mes_auc": STROKE_MIN_AUC,
            "target_mes_auc": STROKE_TARGET_AUC,
            "min_cohen_d": STROKE_MIN_COHEN_D,
            "target_cohen_d": STROKE_TARGET_COHEN_D,
        },
    }
    BENCHMARK_PATH.write_text(json.dumps(bench, indent=2), encoding="utf-8")
    GATES_PATH.write_text(
        json.dumps(
            {
                "min_mes_auc": STROKE_MIN_AUC,
                "min_cohen_d": STROKE_MIN_COHEN_D,
                "target_mes_auc": STROKE_TARGET_AUC,
                "target_cohen_d": STROKE_TARGET_COHEN_D,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"MES AUC={mes_auc:.3f} Cohen d={mes_d:.3f}")
    print("Wrote", BENCHMARK_PATH)

    if args.upload_models:
        import os

        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN required for --upload-models")
        api = HfApi(token=token)
        repo = os.environ.get("HF_MODEL_REPO", "abachu2005/mes-models")
        for name in BUNDLED_MODELS.iterdir():
            api.upload_file(
                path_or_fileobj=str(name),
                path_in_repo=name.name,
                repo_id=repo,
                repo_type="model",
                commit_message="stroke riemannian + weights",
            )

    if mes_auc < STROKE_MIN_AUC or mes_d < STROKE_MIN_COHEN_D:
        print("WARN: below minimum regression gates (improve with full N=50 preprocess)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
