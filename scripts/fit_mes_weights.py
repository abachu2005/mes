#!/usr/bin/env python3
"""Fit MES weights from processed parquet and write mes_core/data bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mes_core.artifacts import weights_bundle_path, write_weights_bundle
from mes_core.config import OPENBCI_MONTAGE_16, TARGET_SFREQ
from mes_core.eval.parquet import download_processed_cache, load_parquet_dir
from mes_core.models.inference import resolve_session_posterior
from mes_core.scoring import (
    MesWeights,
    SubjectBaseline,
    compute_mes,
    fit_mes_weights,
    fit_subject_baseline,
)


def fit_from_trials(
    data_dir: Path, *, max_files: int | None = 60, cohort: str = "healthy"
) -> None:
    prefix = "liu2024_" if cohort == "stroke" else "physionet_"
    trials = load_parquet_dir(
        data_dir,
        prefix=prefix,
        labels={"right_hand", "rest", "break"},
        max_files=max_files,
    )
    if len(trials) < 30 and cohort == "stroke":
        print("Stroke parquet sparse; falling back to physionet for weight fit")
        trials = load_parquet_dir(
            data_dir, prefix="physionet_", labels={"right_hand", "rest"}, max_files=max_files
        )
    if len(trials) < 50:
        raise SystemExit(f"Need >=50 trials, got {len(trials)}")

    ch = list(OPENBCI_MONTAGE_16)
    sfreq = TARGET_SFREQ

    by_subj_rest: dict[str, list[np.ndarray]] = {}
    for t in trials:
        if t.label in ("rest", "break"):
            by_subj_rest.setdefault(t.subject, []).append(t.X)

    z_rows: list[np.ndarray] = []
    p_rows: list[float] = []
    y_rows: list[int] = []

    for t in trials:
        if t.label not in ("right_hand", "rest", "break"):
            continue
        rest_epochs = by_subj_rest.get(t.subject)
        rest_stack = np.stack(rest_epochs) if rest_epochs and len(rest_epochs) >= 2 else None
        if rest_stack is not None:
            bl = fit_subject_baseline(rest_stack, sfreq, ch, "right_hand")
        else:
            bl = SubjectBaseline.zeros(4)

        x = t.X[None, ...]
        try:
            p, _ = resolve_session_posterior(x, "right_hand", cohort=cohort)
            p_val = float(p[0])
        except Exception:
            p_val = 0.5

        res = compute_mes(
            x,
            sfreq,
            ch,
            "right_hand",
            bl,
            MesWeights.default(),
            np.array([p_val]),
            rest_epochs_data=rest_stack,
        )
        z_rows.append(res.z_features[0])
        p_rows.append(p_val)
        y_rows.append(t.y)

    z = np.stack(z_rows)
    p = np.asarray(p_rows)
    y = np.asarray(y_rows)
    weights = fit_mes_weights(z, p, y)

    # Population baseline from all rest trials
    rest_only = [t.X[None, ...] for t in trials if t.label in ("rest", "break")]
    if len(rest_only) >= 5:
        pop_bl = fit_subject_baseline(np.concatenate(rest_only), sfreq, ch, "right_hand")
    else:
        pop_bl = SubjectBaseline.zeros(4)

    out = (
        Path(__file__).resolve().parents[1]
        / "mes_core/data/mes_weights_right_hand_stroke.json"
        if cohort == "stroke"
        else weights_bundle_path("right_hand")
    )
    write_weights_bundle(
        out,
        weights,
        population_baseline=pop_bl,
        meta={
            "n_fit_trials": len(y),
            "n_rest_trials": int((y == 0).sum()),
            "cohort": cohort,
        },
    )
    print("Wrote", out)
    print(weights)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, help="Local processed parquet directory")
    ap.add_argument("--download", action="store_true", help="Fetch parquet from HF Hub first")
    ap.add_argument("--max-files", type=int, default=40)
    ap.add_argument("--upload", action="store_true", help="Upload bundle to HF model repo")
    ap.add_argument("--cohort", choices=["healthy", "stroke"], default="healthy")
    args = ap.parse_args()

    if args.download or args.data_dir is None:
        data_dir = download_processed_cache(max_files=args.max_files)
    else:
        data_dir = args.data_dir

    fit_from_trials(data_dir, max_files=args.max_files, cohort=args.cohort)

    if args.upload:
        import os

        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN required for --upload")
        bundle = weights_bundle_path()
        api = HfApi(token=token)
        repo = os.environ.get("HF_MODEL_REPO", "abachu2005/mes-models")
        api.upload_file(
            path_or_fileobj=str(bundle),
            path_in_repo=bundle.name,
            repo_id=repo,
            repo_type="model",
            commit_message="fit MES weights bundle",
        )
        print("Uploaded to", repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
