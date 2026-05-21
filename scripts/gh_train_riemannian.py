#!/usr/bin/env python3
"""Train Riemannian + logistic regression from processed parquet (GitHub Actions)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

TARGET_T = 750
TARGET_CLASSES = {"right_hand": 1, "rest": 0}


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


def train(data_dir: Path, out_dir: Path) -> None:
    files = sorted(data_dir.glob("*.parquet"))
    train_rows = load_subset(files, "physionet_")
    if len(train_rows) < 20:
        raise RuntimeError(f"Need >=20 trials, got {len(train_rows)}")

    x = np.stack([r["X"] for r in train_rows])
    y = np.array([r["y"] for r in train_rows])
    subjects = np.array([r["subject"] for r in train_rows])
    uniq, counts = np.unique(y, return_counts=True)
    print("X shape:", x.shape, "y dist:", dict(zip(uniq, counts, strict=False)))

    def build_pipe():
        return Pipeline(
            [
                ("cov", Covariances(estimator="oas")),
                ("tan", TangentSpace(metric="riemann")),
                ("lr", LogisticRegression(C=1.0, max_iter=1000)),
            ]
        )

    gkf = GroupKFold(n_splits=min(5, len(np.unique(subjects))))
    accs: list[float] = []
    for tr, te in gkf.split(x, y, groups=subjects):
        pipe = build_pipe()
        pipe.fit(x[tr], y[tr])
        accs.append(float(pipe.score(x[te], y[te])))
        print(f"  fold acc = {accs[-1]:.3f}")
    mean_acc = float(np.mean(accs))
    print(f"LOSO mean acc: {mean_acc:.3f}")

    final = CalibratedClassifierCV(build_pipe(), method="sigmoid", cv=3)
    final.fit(x, y)

    cov_tr = Covariances(estimator="oas").fit(x)
    ts = TangentSpace(metric="riemann").fit(cov_tr.transform(x))
    ts_x = ts.transform(cov_tr.transform(x))
    post_pipe = Pipeline([("lr", LogisticRegression(C=1.0, max_iter=1000))])
    post_pipe.fit(ts_x, y)
    final_post = CalibratedClassifierCV(post_pipe, method="sigmoid", cv=3).fit(ts_x, y)

    onx = to_onnx(
        final_post,
        initial_types=[("X", FloatTensorType([None, ts_x.shape[1]]))],
        target_opset={"": 17, "ai.onnx.ml": 3},
        options={id(final_post): {"zipmap": False}},
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "riemannian_lr_right_hand.onnx"
    onnx_path.write_bytes(onx.SerializeToString())
    meta = {
        "model": "riemannian_lr",
        "task": "right_hand_vs_rest",
        "input_kind": "tangent_space",
        "n_features": int(ts_x.shape[1]),
        "n_classes": 2,
        "loso_acc": mean_acc,
        "n_train": int(x.shape[0]),
        "channels": 16,
        "sfreq": 125.0,
        "epoch_samples": int(x.shape[2]),
        "trained_on": "github_actions",
    }
    (out_dir / "riemannian_lr_right_hand.json").write_text(json.dumps(meta, indent=2))
    cov_est = getattr(cov_tr, "estimator_", None) or cov_tr.estimator
    np.savez(
        str(out_dir / "riemannian_lr_frontend.npz"),
        cov_estimator=np.array([cov_est], dtype=object),
        ts_reference=ts.reference_,
    )
    print("Wrote", onnx_path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("/tmp/mes-models"))
    args = p.parse_args()
    train(args.data, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
