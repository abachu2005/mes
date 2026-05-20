#!/usr/bin/env python3
"""Download PhysioNet eegmmidb, preprocess, write parquet for MES training.

Runs on GitHub Actions (reliable DNS). Kaggle kernels cannot reach physionet.org.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci

from mes_core.config import EPOCH_TMAX, EPOCH_TMIN, TARGET_SFREQ
from mes_core.preprocessing import PreprocessConfig, epoch_raw, preprocess_raw

MI_HAND_RUNS = [4, 8, 12]
REST_RUNS = [1]


def load_pn_raws(subject: int, runs: list[int]):
    files = eegbci.load_data(subject, runs, update_path=True, verbose="ERROR")
    raws = [mne.io.read_raw_edf(f, preload=True, verbose="ERROR") for f in files]
    raw = mne.concatenate_raws(raws, verbose="ERROR")
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
    return raw


def epochs_to_parquet(epochs, dataset_name: str, subject_id: str, label_map: dict[str, int], out_dir: Path):
    if epochs is None or len(epochs) == 0:
        return None
    data = epochs.get_data()
    ev = epochs.events[:, 2]
    rev = {v: k for k, v in label_map.items()}
    labels = pd.Series(ev).map(rev).fillna("unknown").tolist()
    rows = []
    for i, (arr, lab) in enumerate(zip(data, labels, strict=False)):
        rows.append(
            {
                "dataset": dataset_name,
                "subject": str(subject_id),
                "trial": i,
                "label": lab,
                "sfreq": float(epochs.info["sfreq"]),
                "ch_names": json.dumps(epochs.info["ch_names"]),
                "data": arr.astype("float32").tobytes(),
                "n_channels": int(arr.shape[0]),
                "n_times": int(arr.shape[1]),
            }
        )
    df = pd.DataFrame(rows)
    out = out_dir / f"{dataset_name}_S{subject_id}.parquet"
    df.to_parquet(out, compression="zstd")
    return out


def write_rest_parquet(raw_pp, sid: int, out_dir: Path) -> None:
    arr = raw_pp.get_data()
    sfreq = raw_pp.info["sfreq"]
    ch_names = raw_pp.info["ch_names"]
    n_win = int((EPOCH_TMAX - EPOCH_TMIN) * sfreq)
    n_trials = arr.shape[1] // n_win
    if n_trials <= 0:
        return
    rows = []
    for i in range(n_trials):
        a = arr[:, i * n_win : (i + 1) * n_win]
        rows.append(
            {
                "dataset": "physionet",
                "subject": f"{sid}_rest",
                "trial": i,
                "label": "rest",
                "sfreq": float(sfreq),
                "ch_names": json.dumps(ch_names),
                "data": a.astype("float32").tobytes(),
                "n_channels": int(a.shape[0]),
                "n_times": int(a.shape[1]),
            }
        )
    pd.DataFrame(rows).to_parquet(out_dir / f"physionet_S{sid}_rest.parquet", compression="zstd")


def run(out_dir: Path, n_subjects: int) -> dict:
    mne.set_log_level("ERROR")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PreprocessConfig(do_ica=False)

    print(f"=== PhysioNet eegmmidb — {n_subjects} subjects ===")
    for sid in range(1, n_subjects + 1):
        try:
            raw_mi = load_pn_raws(sid, MI_HAND_RUNS)
            raw_mi_pp = preprocess_raw(raw_mi, cfg)
            events, ev_id = mne.events_from_annotations(raw_mi_pp, verbose="ERROR")
            wanted: dict[str, int] = {}
            if "T1" in ev_id:
                wanted["left_hand"] = ev_id["T1"]
            if "T2" in ev_id:
                wanted["right_hand"] = ev_id["T2"]
            if wanted:
                ep = epoch_raw(raw_mi_pp, event_id=wanted)
                if ep is not None and len(ep) > 0:
                    epochs_to_parquet(ep, "physionet", f"{sid}_mi", wanted, out_dir)

            try:
                raw_rest = load_pn_raws(sid, REST_RUNS)
                write_rest_parquet(preprocess_raw(raw_rest, cfg), sid, out_dir)
            except Exception as e:
                print(f"  subject {sid} rest FAIL: {e}")

            if sid % 5 == 0:
                print(f"  subject {sid}/{n_subjects} done ({len(list(out_dir.glob('*.parquet')))} files)")
        except Exception as e:
            print(f"  subject {sid} FAIL: {e}")
        gc.collect()

    files = sorted(p.name for p in out_dir.glob("*.parquet"))
    manifest = {"stage": "preprocess", "n_parquet": len(files), "files": files, "runner": "github_actions"}
    manifest_path = out_dir.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest['n_parquet']} parquet files to {out_dir}")
    if manifest["n_parquet"] == 0:
        raise RuntimeError("preprocess produced zero parquet files — check PhysioNet download logs")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("/tmp/mes-out/processed"))
    p.add_argument("--subjects", type=int, default=30)
    args = p.parse_args()
    run(args.out, args.subjects)
    return 0


if __name__ == "__main__":
    sys.exit(main())
