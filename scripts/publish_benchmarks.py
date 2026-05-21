#!/usr/bin/env python3
"""Publish benchmarks.md/json to the HF model repo from training sidecars.

Uses loso metrics in ``*_right_hand.json`` files already uploaded with ONNX.
Full held-out evaluation still comes from ``notebooks/kaggle/03_validate.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _repo() -> str:
    user = os.environ.get("HF_USERNAME", "abachu2005")
    return os.environ.get("HF_MODEL_REPO", f"{user}/mes-models")


def _token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise SystemExit("HF_TOKEN is required to upload benchmarks")
    return tok


def build_benchmarks() -> tuple[dict, str]:
    from huggingface_hub import hf_hub_download

    repo = _repo()
    token = os.environ.get("HF_TOKEN")
    sidecars = {
        "riemannian_lr_right_hand.onnx": "riemannian_lr_right_hand.json",
        "eegnet_right_hand.onnx": "eegnet_right_hand.json",
    }
    rows: list[dict] = []
    for onnx_name, json_name in sidecars.items():
        try:
            p = Path(
                hf_hub_download(
                    repo_id=repo,
                    filename=json_name,
                    repo_type="model",
                    token=token,
                )
            )
            meta = json.loads(p.read_text())
        except Exception as e:
            print(f"WARN: skip {json_name}: {e}", file=sys.stderr)
            continue
        acc = meta.get("loso_acc_mean") or meta.get("loso_acc")
        rows.append(
            {
                "model_file": onnx_name,
                "model_id": meta.get("model", onnx_name),
                "task": meta.get("task", ""),
                "n_train": meta.get("n_train"),
                "loso_accuracy": float(acc) if acc is not None else None,
                "trained_on": meta.get("trained_on", ""),
                "extra": {k: meta[k] for k in ("fold_accs", "n_channels", "n_times") if k in meta},
            }
        )

    payload = {
        "source": "training_sidecars",
        "dataset": os.environ.get("HF_DATASET_REPO", "abachu2005/mes-eeg-processed"),
        "models": rows,
    }
    md = [
        "# MES Benchmarks\n",
        "\nAuto-generated from training sidecar JSON on the model repo "
        "(LOSO cross-validation during train). "
        "For per-dataset held-out metrics, run `notebooks/kaggle/03_validate.py`.\n",
        "\n| Model | Task | n_train | LOSO accuracy | Trained on |\n",
        "|---|---|---:|---:|---|\n",
    ]
    for r in rows:
        acc = r["loso_accuracy"]
        acc_s = f"{acc:.3f}" if acc is not None else "—"
        n = r["n_train"] if r["n_train"] is not None else "—"
        md.append(
            f"| `{r['model_file']}` | {r['task']} | {n} | {acc_s} | {r['trained_on']} |\n"
        )
    md.append(
        "\n**Inference:** When both ONNX files are present, the API averages "
        "Riemannian tangent-space and EEGNet posteriors (`ensemble(...)`).\n"
    )
    return payload, "".join(md)


def upload(payload: dict, md: str) -> None:
    from huggingface_hub import HfApi

    repo = _repo()
    token = _token()
    api = HfApi(token=token)
    out = ROOT / ".benchmarks_out"
    out.mkdir(exist_ok=True)
    json_path = out / "benchmarks.json"
    md_path = out / "benchmarks.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(md)
    for path in (json_path, md_path):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo,
            repo_type="model",
            commit_message="publish benchmarks from training sidecars",
        )
    print(f"OK: https://huggingface.co/{repo}/blob/main/benchmarks.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print artifacts only")
    args = ap.parse_args()
    payload, md = build_benchmarks()
    if not payload["models"]:
        print("No model sidecars found; nothing to publish", file=sys.stderr)
        return 1
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        print(md)
        return 0
    upload(payload, md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
