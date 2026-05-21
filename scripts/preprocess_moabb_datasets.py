#!/usr/bin/env python3
"""Preprocess additional MOABB datasets (BCI IV 2b, Lee2019 stroke MI) to processed parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def preprocess_bci_iv_2b(out_dir: Path, *, subjects: list[int] | None = None) -> int:
    """Export BCI Competition IV 2b subjects to MES parquet format."""
    import mne
    import numpy as np
    import pandas as pd

    try:
        from moabb.datasets import BNCI2014_001
        ds = BNCI2014_001()
    except Exception as e:
        print("MOABB BCI IV dataset unavailable:", e)
        return 0

    from mes_core.preprocessing import PreprocessConfig, preprocess_raw
    from scripts.gh_preprocess_physionet import epochs_to_parquet, write_rest_parquet

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PreprocessConfig()
    n_files = 0
    subjs = subjects or list(range(1, 10))
    for sid in subjs:
        try:
            raw_dict = ds.get_data(subjects=[sid])
        except Exception as e:
            print(f"skip subject {sid}: {e}")
            continue
        for session, raw in raw_dict.items():
            if not hasattr(raw, "info"):
                continue
            raw_pp = preprocess_raw(raw, cfg)
            # MOABB events vary — export windows as MI if possible
            events, event_id = mne.events_from_annotations(raw_pp, verbose="ERROR")
            epochs = mne.Epochs(
                raw_pp, events, event_id=event_id, tmin=-2, tmax=4, baseline=(-1.5, -0.5),
                preload=True, verbose="ERROR",
            )
            label_map = {k: i for i, k in enumerate(event_id)}
            epochs_to_parquet(
                epochs, "bci_iv_2b", f"S{sid}_{session}", label_map, out_dir
            )
            n_files += 1
    print(f"BCI IV 2b: wrote {n_files} epoch sets to {out_dir}")
    return n_files


def preprocess_lee2019_stroke(out_dir: Path, *, max_subjects: int = 20) -> int:
    """Lee2019_MI stroke dataset via MOABB when installed."""
    try:
        from moabb.datasets import Lee2019_MI
    except ImportError:
        print("Lee2019_MI not in this MOABB version — skip stroke preprocess")
        return 0

    from mes_core.preprocessing import PreprocessConfig, preprocess_raw
    from scripts.gh_preprocess_physionet import epochs_to_parquet

    ds = Lee2019_MI()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PreprocessConfig()
    n = 0
    for sid in ds.subject_list[:max_subjects]:
        try:
            raw_dict = ds.get_data(subjects=[sid])
        except Exception as e:
            print(f"lee skip {sid}: {e}")
            continue
        for session, raw in raw_dict.items():
            import mne

            raw_pp = preprocess_raw(raw, cfg)
            events, event_id = mne.events_from_annotations(raw_pp, verbose="ERROR")
            epochs = mne.Epochs(
                raw_pp, events, event_id=event_id, tmin=-2, tmax=4, baseline=(-1.5, -0.5),
                preload=True, verbose="ERROR",
            )
            rev = {v: k for k, v in event_id.items()}
            label_map = {}
            for k in event_id:
                kl = k.lower()
                if "right" in kl:
                    label_map[k] = 1
                elif "rest" in kl or "idle" in kl:
                    label_map[k] = 0
            if not label_map:
                continue
            epochs_to_parquet(epochs, "liu2024", f"S{sid}", label_map, out_dir)
            n += 1
    print(f"Liu/stroke MOABB: {n} files")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/mes-out/processed"))
    ap.add_argument("--bci-iv-2b", action="store_true")
    ap.add_argument("--stroke", action="store_true")
    args = ap.parse_args()
    total = 0
    if args.bci_iv_2b:
        total += preprocess_bci_iv_2b(args.out)
    if args.stroke:
        total += preprocess_lee2019_stroke(args.out)
    if not args.bci_iv_2b and not args.stroke:
        total += preprocess_bci_iv_2b(args.out)
        total += preprocess_lee2019_stroke(args.out)
    return 0 if total >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
