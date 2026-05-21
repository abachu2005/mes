#!/usr/bin/env python3
"""Publish Liu stroke parquet, models, clinical TSV, and benchmarks to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        raise SystemExit("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) required")
    return tok


def _repos() -> tuple[str, str]:
    user = os.environ.get("HF_USERNAME", "abachu2005")
    ds = os.environ.get("HF_DATASET_REPO", f"{user}/mes-eeg-processed")
    mdl = os.environ.get("HF_MODEL_REPO", f"{user}/mes-models")
    return ds, mdl


def upload_parquet(data_dir: Path, dataset_repo: str) -> int:
    from huggingface_hub import HfApi, create_repo

    files = sorted(data_dir.glob("liu2024_*.parquet")) + sorted(data_dir.glob("liu2025_*.parquet"))
    if not files:
        print("No stroke parquet to upload", file=sys.stderr)
        return 0
    token = _token()
    api = HfApi(token=token)
    create_repo(repo_id=dataset_repo, repo_type="dataset", exist_ok=True, token=token)
    n = 0
    for f in files:
        path_in_repo = f"processed/{f.name}"
        print("upload", path_in_repo)
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=path_in_repo,
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message="Liu2024/2025 stroke processed parquet",
        )
        n += 1
    print(f"Uploaded {n} parquet files to {dataset_repo}")
    return n


def upload_models(model_dir: Path, model_repo: str) -> int:
    from huggingface_hub import HfApi, create_repo

    patterns = ("*.onnx", "*.json", "*.npz")
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(model_dir.glob(pat)))
    if not files:
        print("No model artifacts in", model_dir, file=sys.stderr)
        return 0
    token = _token()
    api = HfApi(token=token)
    create_repo(repo_id=model_repo, repo_type="model", exist_ok=True, token=token)
    for f in files:
        print("upload model", f.name)
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=model_repo,
            repo_type="model",
            commit_message="stroke production models",
        )
    return len(files)


def upload_clinical_tsv(tsv: Path, dataset_repo: str) -> None:
    if not tsv.exists():
        print("Skip clinical TSV (missing):", tsv)
        return
    from huggingface_hub import HfApi, create_repo

    token = _token()
    api = HfApi(token=token)
    create_repo(repo_id=dataset_repo, repo_type="dataset", exist_ok=True, token=token)
    api.upload_file(
        path_or_fileobj=str(tsv),
        path_in_repo="clinical/liu2024_clinical.tsv",
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message="Liu2024 per-subject clinical table",
    )
    print("Uploaded clinical TSV")


def upload_benchmarks(model_repo: str) -> None:
    bench = ROOT / "mes_core/data/benchmarks.json"
    gates = ROOT / "mes_core/data/stroke_validation_gates.json"
    if not bench.exists():
        print("Skip benchmarks (missing)")
        return
    from huggingface_hub import HfApi, create_repo

    token = _token()
    api = HfApi(token=token)
    create_repo(repo_id=model_repo, repo_type="model", exist_ok=True, token=token)
    payload = json.loads(bench.read_text())
    if gates.exists():
        payload["regression_gates"] = json.loads(gates.read_text())
    md = [
        "# MES Benchmarks\n\n",
        "Stroke Liu2024 held-out metrics from `scripts/run_stroke_pipeline.py`.\n\n",
        "## Stroke (Liu2024)\n\n",
    ]
    s = payload.get("stroke_liu2024") or {}
    md.append(f"- Trials: {s.get('n_trials')}\n")
    md.append(f"- Subjects: {s.get('n_subjects')}\n")
    md.append(f"- MES AUC: {s.get('mes_auc', 0):.3f}\n")
    md.append(f"- MES Cohen's d: {s.get('mes_cohen_d', 0):.3f}\n")
    md.append(f"- Riemannian LOSO acc: {s.get('riemannian_loso_acc', 0):.3f}\n\n")
    for m in s.get("models") or []:
        md.append(f"- {m.get('name')}: AUC {m.get('auc', 0):.3f}\n")

    out = ROOT / ".benchmarks_out"
    out.mkdir(exist_ok=True)
    (out / "benchmarks.json").write_text(json.dumps(payload, indent=2))
    (out / "benchmarks.md").write_text("".join(md))
    for p in (out / "benchmarks.json", out / "benchmarks.md"):
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=p.name,
            repo_id=model_repo,
            repo_type="model",
            commit_message="stroke validation benchmarks",
        )
    print("Uploaded benchmarks.md/json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/processed_stroke")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "mes_core/data/models")
    ap.add_argument("--clinical-tsv", type=Path, default=ROOT / "mes_core/data/liu2024_clinical.tsv")
    ap.add_argument("--skip-parquet", action="store_true")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-benchmarks", action="store_true")
    args = ap.parse_args()
    ds_repo, mdl_repo = _repos()
    if not args.skip_parquet:
        upload_parquet(args.data_dir, ds_repo)
    if not args.skip_models:
        upload_models(args.model_dir, mdl_repo)
    upload_clinical_tsv(args.clinical_tsv, ds_repo)
    if not args.skip_benchmarks:
        upload_benchmarks(mdl_repo)
    print("Done:", f"https://huggingface.co/datasets/{ds_repo}", f"https://huggingface.co/{mdl_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
