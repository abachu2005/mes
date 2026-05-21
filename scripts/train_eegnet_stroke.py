#!/usr/bin/env python3
"""Train EEGNet on local Liu2024 parquet and bundle stroke ONNX for production."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# braindecode 0.8.x expects legacy MOABB class names
import moabb.datasets as _moabb_ds  # noqa: E402

if not hasattr(_moabb_ds, "BNCI2014001"):
    _moabb_ds.BNCI2014001 = _moabb_ds.BNCI2014_001  # type: ignore[attr-defined]

from braindecode.models import EEGNetv4  # noqa: E402
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TARGET_T = 750
N_CH = 16
STROKE_LABELS = {"right_hand": 1, "rest": 0, "break": 0}
BUNDLED = ROOT / "mes_core/data/models"


def load_stroke_rows(data_dir: Path, prefix: str = "liu2024_") -> list[dict]:
    rows: list[dict] = []
    for f in sorted(data_dir.glob(f"{prefix}*.parquet")):
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            lab = str(r["label"])
            if lab not in STROKE_LABELS:
                continue
            x = np.frombuffer(r["data"], dtype="float32").reshape(r["n_channels"], r["n_times"])
            n = x.shape[1]
            if n >= TARGET_T:
                x = x[:, :TARGET_T]
            else:
                x = np.concatenate([x, np.zeros((x.shape[0], TARGET_T - n), dtype="float32")], 1)
            rows.append({"X": x, "y": STROKE_LABELS[lab], "subject": r["subject"]})
    return rows


def train(
    data_dir: Path,
    out_dir: Path,
    *,
    n_epochs: int = 50,
    prefix: str = "liu2024_",
) -> dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_stroke_rows(data_dir, prefix=prefix)
    if len(rows) < 50:
        raise RuntimeError(f"Need >=50 trials, got {len(rows)}")

    x = np.stack([r["X"] for r in rows])
    y = np.array([r["y"] for r in rows])
    subj = np.array([r["subject"] for r in rows])
    print("device:", device, "X:", x.shape, "y:", np.unique(y, return_counts=True))

    def make_model() -> EEGNetv4:
        return EEGNetv4(n_chans=N_CH, n_outputs=2, n_times=TARGET_T, drop_prob=0.5).to(device)

    def train_fold(xtr, ytr, xte, yte) -> tuple[EEGNetv4, float]:
        xt = torch.tensor(xtr, dtype=torch.float32)
        yt = torch.tensor(ytr, dtype=torch.long)
        xv = torch.tensor(xte, dtype=torch.float32)
        dl = DataLoader(TensorDataset(xt, yt), batch_size=64, shuffle=True)
        model = make_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        best, bad = 0.0, 0
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
            acc = float((pred == yte).mean())
            if acc > best:
                best, bad = acc, 0
            else:
                bad += 1
                if bad >= 8:
                    break
        model.eval()
        with torch.no_grad():
            pred = model(xv.to(device)).argmax(1).cpu().numpy()
        return model, float((pred == yte).mean())

    gkf = GroupKFold(n_splits=min(5, len(np.unique(subj))))
    accs: list[float] = []
    for i, (tr, te) in enumerate(gkf.split(x, y, groups=subj)):
        _, acc = train_fold(x[tr], y[tr], x[te], y[te])
        accs.append(acc)
        print(f"  fold {i}: {acc:.3f}")

    final = make_model()
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    dl = DataLoader(TensorDataset(xt, yt), batch_size=64, shuffle=True)
    opt = torch.optim.Adam(final.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        final.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(final(xb), yb).backward()
            opt.step()

    out_dir.mkdir(parents=True, exist_ok=True)
    final.eval().cpu()
    dummy = torch.zeros(1, N_CH, TARGET_T)
    onnx_path = out_dir / "eegnet_right_hand_stroke.onnx"
    export_kw = {
        "input_names": ["X"],
        "output_names": ["proba"],
        "dynamic_axes": {"X": {0: "batch"}, "proba": {0: "batch"}},
        "opset_version": 17,
    }
    try:
        torch.onnx.export(final, dummy, str(onnx_path), dynamo=False, **export_kw)
    except TypeError:
        torch.onnx.export(final, dummy, str(onnx_path), **export_kw)

    meta = {
        "model": "eegnet_v4",
        "task": "right_hand_vs_rest",
        "cohort": "stroke",
        "input_kind": "raw_epochs",
        "n_channels": N_CH,
        "n_times": TARGET_T,
        "loso_acc_mean": float(np.mean(accs)),
        "fold_accs": accs,
        "n_train": int(x.shape[0]),
        "data_prefix": prefix,
    }
    json_path = out_dir / "eegnet_right_hand_stroke.json"
    json_path.write_text(json.dumps(meta, indent=2))
    print("Wrote", onnx_path)
    return {"loso_acc_mean": float(np.mean(accs)), "n_train": int(x.shape[0])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/processed_stroke")
    ap.add_argument("--out", type=Path, default=Path("/tmp/mes-eegnet-stroke"))
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    stats = train(args.data_dir, args.out, n_epochs=args.epochs)
    BUNDLED.mkdir(parents=True, exist_ok=True)
    for name in ("eegnet_right_hand_stroke.onnx", "eegnet_right_hand_stroke.json"):
        src = args.out / name
        if src.exists():
            shutil.copy2(src, BUNDLED / name)
    print("Bundled to", BUNDLED, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
