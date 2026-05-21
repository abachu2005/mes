# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26,<2.2",
#   "pandas>=2.1,<3",
#   "pyarrow>=15",
#   "torch>=2.2,<3",
#   "braindecode>=0.8,<1",
#   "onnx>=1.16,<2",
#   "onnxscript>=0.1",
#   "scikit-learn>=1.4,<2",
#   "huggingface_hub>=0.24,<1",
# ]
# ///
"""Train EEGNet v4 on HF Hub processed parquet and publish ONNX to mes-models.

Designed for `hf jobs uv run --flavor a10g-small` (GPU, ~1–2 h, no Kaggle queue).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from braindecode.models import EEGNetv4
from huggingface_hub import HfApi, create_repo, snapshot_download
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TARGET_T = 750  # 6 s @ 125 Hz
N_CH = 16
TARGET_CLASSES = {"right_hand": 1, "rest": 0}
OUT = Path(os.environ.get("MES_TRAIN_OUT", "/tmp/mes-eegnet"))


def _repo_ids() -> tuple[str, str]:
    user = os.environ.get("HF_USERNAME", "abachu2005")
    ds = os.environ.get("HF_DATASET_REPO", f"{user}/mes-eeg-processed")
    mdl = os.environ.get("HF_MODEL_REPO", f"{user}/mes-models")
    return ds, mdl


def _token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN missing")
    return tok


def download_parquet_dir() -> Path:
    ds_repo, _ = _repo_ids()
    token = _token()
    print(f"Downloading dataset {ds_repo} ...")
    root = Path(snapshot_download(repo_id=ds_repo, repo_type="dataset", token=token))
    for candidate in (root / "processed", root):
        if candidate.is_dir() and list(candidate.glob("*.parquet")):
            print(f"Using parquet dir: {candidate}")
            return candidate
    raise FileNotFoundError(f"No parquet under {root}")


def load_subset(files: list[Path], prefix: str) -> list[dict]:
    rows: list[dict] = []
    for f in files:
        if not f.name.startswith(prefix):
            continue
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            if r["label"] not in TARGET_CLASSES:
                continue
            x = np.frombuffer(r["data"], dtype="float32").reshape(r["n_channels"], r["n_times"])
            n = x.shape[1]
            if n >= TARGET_T:
                x = x[:, :TARGET_T]
            else:
                x = np.concatenate([x, np.zeros((x.shape[0], TARGET_T - n), dtype="float32")], 1)
            rows.append({"X": x, "y": TARGET_CLASSES[r["label"]], "subject": r["subject"]})
    return rows


def train_eegnet() -> tuple[Path, Path]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    ds_dir = download_parquet_dir()
    files = sorted(ds_dir.glob("*.parquet"))
    train_rows = load_subset(files, "physionet_") or load_subset(files, "bci_iv_2a_")
    if not train_rows:
        raise RuntimeError("No training rows found in processed parquet")

    x = np.stack([r["X"] for r in train_rows])
    y = np.array([r["y"] for r in train_rows])
    subj = np.array([r["subject"] for r in train_rows])
    uniq, counts = np.unique(y, return_counts=True)
    print("X:", x.shape, "y dist:", dict(zip(uniq, counts, strict=False)))

    def make_model(n_classes: int = 2) -> EEGNetv4:
        return EEGNetv4(n_chans=N_CH, n_outputs=n_classes, n_times=TARGET_T, drop_prob=0.5).to(device)

    def train_one_fold(xtr, ytr, xte, yte, n_epochs: int = 80, batch: int = 64, lr: float = 1e-3):
        xt = torch.tensor(xtr, dtype=torch.float32)
        yt = torch.tensor(ytr, dtype=torch.long)
        xv = torch.tensor(xte, dtype=torch.float32)
        yv = torch.tensor(yte, dtype=torch.long)
        dl = DataLoader(TensorDataset(xt, yt), batch_size=batch, shuffle=True)
        model = make_model()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(n_epochs):
            model.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(xv.to(device)).argmax(1).cpu().numpy()
        return model, float((pred == yte).mean())

    gkf = GroupKFold(n_splits=min(5, len(np.unique(subj))))
    fold_accs: list[float] = []
    for i, (tr, te) in enumerate(gkf.split(x, y, groups=subj)):
        _, acc = train_one_fold(x[tr], y[tr], x[te], y[te])
        fold_accs.append(acc)
        print(f"  fold {i}: {acc:.3f}")
    print("LOSO mean acc:", float(np.mean(fold_accs)))

    final_model = make_model()
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    dl = DataLoader(TensorDataset(xt, yt), batch_size=64, shuffle=True)
    opt = torch.optim.Adam(final_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(80):
        final_model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(final_model(xb), yb).backward()
            opt.step()

    OUT.mkdir(parents=True, exist_ok=True)
    final_model.eval()
    dummy = torch.zeros(1, N_CH, TARGET_T, dtype=torch.float32, device=device)
    onnx_path = OUT / "eegnet_right_hand.onnx"
    torch.onnx.export(
        final_model,
        dummy,
        str(onnx_path),
        input_names=["X"],
        output_names=["proba"],
        dynamic_axes={"X": {0: "batch"}, "proba": {0: "batch"}},
        opset_version=17,
    )
    meta = {
        "model": "eegnet_v4",
        "task": "right_hand_vs_rest",
        "input_kind": "raw_epochs",
        "n_channels": N_CH,
        "n_times": TARGET_T,
        "sfreq": 125.0,
        "n_classes": 2,
        "loso_acc_mean": float(np.mean(fold_accs)),
        "fold_accs": fold_accs,
        "n_train": int(x.shape[0]),
        "trained_on": "hf_jobs",
    }
    json_path = OUT / "eegnet_right_hand.json"
    json_path.write_text(json.dumps(meta, indent=2))
    print("Wrote", onnx_path, json_path)
    return onnx_path, json_path


def upload_artifacts(onnx_path: Path, json_path: Path) -> None:
    _, mdl_repo = _repo_ids()
    token = _token()
    api = HfApi(token=token)
    create_repo(repo_id=mdl_repo, repo_type="model", exist_ok=True, token=token)
    for path in (onnx_path, json_path):
        print(f"Uploading {path.name} -> {mdl_repo}")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=mdl_repo,
            repo_type="model",
            commit_message="train eegnet via HF Jobs",
        )
    print(f"OK: https://huggingface.co/{mdl_repo}")


def main() -> int:
    onnx_path, json_path = train_eegnet()
    upload_artifacts(onnx_path, json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
