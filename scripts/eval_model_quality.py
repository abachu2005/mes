#!/usr/bin/env python3
"""Evaluate MES and ONNX models on processed parquet (prints summary table)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mes_core.eval.parquet import download_processed_cache, load_parquet_dir
from mes_core.eval.validate import _trial_features_and_posterior
from mes_core.artifacts import load_population_baseline
from mes_core.models.inference import load_onnx_model, _predict_target_proba, _fit_epoch_window
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


def loso_onnx(trials, filename: str) -> tuple[float, list[float]]:
    """GroupKFold LOSO accuracy for one ONNX file."""
    from sklearn.model_selection import GroupKFold

    try:
        clf = load_onnx_model(filename)
    except Exception as e:
        return float("nan"), [float("nan")]

    X = np.stack([t.X for t in trials])
    y = np.array([t.y for t in trials])
    groups = np.array([t.subject for t in trials])
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    accs = []
    for tr, te in gkf.split(X, y, groups=groups):
        # Simple: fit riemannian on train only for tangent path
        x_tr, x_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        if len(clf.feature_shape) == 1:
            cov = Covariances(estimator="oas").fit_transform(x_tr.astype(float))
            ts_fit = TangentSpace(metric="riemann").fit(cov)
            ts_tr = ts_fit.transform(Covariances(estimator="oas").fit_transform(x_tr.astype(float)))
            ts_te = ts_fit.transform(Covariances(estimator="oas").fit_transform(x_te.astype(float)))
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=1000).fit(ts_tr, y_tr)
            accs.append(float(lr.score(ts_te, y_te)))
        else:
            x_tr = _fit_epoch_window(x_tr)
            x_te = _fit_epoch_window(x_te)
            proba_tr = clf.predict_proba(x_tr.astype(np.float32))
            proba_te = clf.predict_proba(x_te.astype(np.float32))
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=1000).fit(
                proba_tr[:, 1:2] if proba_tr.ndim == 2 else proba_tr.reshape(-1, 1), y_tr
            )
            pred = lr.predict(proba_te[:, 1:2] if proba_te.ndim == 2 else proba_te.reshape(-1, 1))
            accs.append(float((pred.ravel() == y_te).mean()))
    return float(np.mean(accs)), accs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--max-files", type=int, default=60)
    args = ap.parse_args()

    data_dir = args.data_dir or (
        download_processed_cache(max_files=args.max_files) if args.download else None
    )
    if data_dir is None:
        raise SystemExit("Provide --data-dir or --download")

    trials = load_parquet_dir(data_dir, prefix="physionet_", max_files=args.max_files)
    rh = [t for t in trials if t.label in ("right_hand", "rest")]
    print(f"Trials: {len(rh)} (right_hand={sum(t.y==1 for t in rh)}, rest={sum(t.y==0 for t in rh)})")
    print(f"Subjects: {len({t.subject for t in rh})}")

    pop = load_population_baseline()
    z, p, labels, mes = _trial_features_and_posterior(rh, population_baseline=pop)

    task_med = float(np.median(mes[labels == 1]))
    pred_mes = (mes >= task_med).astype(int)
    pred_p = (p >= 0.5).astype(int)

    results = {
        "n_trials": len(rh),
        "ensemble_posterior": {
            "accuracy": float(accuracy_score(labels, pred_p)),
            "auc": float(roc_auc_score(labels, p)),
            "mean_p_task": float(p[labels == 1].mean()),
            "mean_p_rest": float(p[labels == 0].mean()),
        },
        "mes_score": {
            "accuracy": float(accuracy_score(labels, pred_mes)),
            "auc": float(roc_auc_score(labels, mes / 100.0)),
            "mean_mes_task": float(mes[labels == 1].mean()),
            "mean_mes_rest": float(mes[labels == 0].mean()),
        },
    }

    for name in ("riemannian_lr_right_hand.onnx", "eegnet_right_hand.onnx"):
        mean_acc, folds = loso_onnx(rh, name)
        results[name] = {"loso_mean_accuracy": mean_acc, "fold_accs": folds}

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
