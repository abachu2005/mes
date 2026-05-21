#!/usr/bin/env python3
"""Preprocess MOABB datasets (BCI IV 2b, Liu2024/Liu2025 stroke MI) to MES parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _iter_moabb_raws(raw_dict: dict) -> Iterator[tuple[int | str, Any, Any, Any]]:
    """Yield (subject_id, session_id, run_id, mne.io.Raw) from MOABB get_data()."""
    for sub_id, sessions in raw_dict.items():
        if hasattr(sessions, "info"):
            yield sub_id, 0, 0, sessions
            continue
        for sess_id, runs in sessions.items():
            if hasattr(runs, "info"):
                yield sub_id, sess_id, 0, runs
                continue
            for run_id, raw in runs.items():
                if hasattr(raw, "info"):
                    yield sub_id, sess_id, run_id, raw


def preprocess_bci_iv_2b(out_dir: Path, *, subjects: list[int] | None = None) -> int:
    """Export BCI Competition IV 2b subjects to MES parquet format."""
    import mne

    try:
        from moabb.datasets import BNCI2014_001

        ds = BNCI2014_001()
    except Exception as e:
        print("MOABB BCI IV dataset unavailable:", e)
        return 0

    from mes_core.preprocessing import PreprocessConfig, preprocess_raw
    from scripts.gh_preprocess_physionet import epochs_to_parquet

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
        for _sub, _sess, _run, raw in _iter_moabb_raws(raw_dict):
            raw_pp = preprocess_raw(raw, cfg)
            events, event_id = mne.events_from_annotations(raw_pp, verbose="ERROR")
            epochs = mne.Epochs(
                raw_pp,
                events,
                event_id=event_id,
                tmin=-2,
                tmax=4,
                baseline=(-1.5, -0.5),
                preload=True,
                verbose="ERROR",
            )
            label_map = {k: i for i, k in enumerate(event_id)}
            epochs_to_parquet(epochs, "bci_iv_2b", f"S{sid}", label_map, out_dir)
            n_files += 1
    print(f"BCI IV 2b: wrote {n_files} epoch sets to {out_dir}")
    return n_files


def _export_mi_epochs(
    raw_dict: dict,
    *,
    out_dir: Path,
    dataset_prefix: str,
    subject_tag: str,
    cfg: Any,
    label_map: dict[str, int],
) -> int:
    import mne

    from mes_core.preprocessing import preprocess_raw
    from scripts.gh_preprocess_physionet import epochs_to_parquet

    n = 0
    for sub_id, sess_id, run_id, raw in _iter_moabb_raws(raw_dict):
        raw_pp = preprocess_raw(raw, cfg)
        events, event_id = mne.events_from_annotations(raw_pp, verbose="ERROR")
        # Keep only events we can label.
        filt = {k: v for k, v in label_map.items() if k in event_id}
        if not filt:
            continue
        epochs = mne.Epochs(
            raw_pp,
            events,
            event_id=filt,
            tmin=-2,
            tmax=4,
            baseline=(-1.5, -0.5),
            preload=True,
            verbose="ERROR",
        )
        sid = f"{subject_tag}{sub_id}_s{sess_id}_r{run_id}"
        epochs_to_parquet(epochs, dataset_prefix, sid, filt, out_dir)
        n += 1
    return n


def preprocess_liu2024(out_dir: Path, *, max_subjects: int = 50) -> int:
    """Acute stroke hand MI — Liu et al. 2024 (Sci Data), via MOABB."""
    try:
        from moabb.datasets import Liu2024
    except ImportError:
        print("Liu2024 not in this MOABB version — pip install -U moabb")
        return 0

    from mes_core.preprocessing import PreprocessConfig, preprocess_raw

    ds = Liu2024(break_events=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PreprocessConfig()
    n = 0
    for sid in ds.subject_list[:max_subjects]:
        try:
            raw_dict = ds.get_data(subjects=[sid])
        except Exception as e:
            print(f"liu2024 skip {sid}: {e}")
            continue
        import mne

        for sub_id, sess_id, run_id, raw in _iter_moabb_raws(raw_dict):
            raw_pp = preprocess_raw(raw, cfg)
            _events, event_id = mne.events_from_annotations(raw_pp, verbose="ERROR")
            label_map = {
                k: int(v) for k, v in event_id.items() if k in ("right_hand", "break")
            }
            if "right_hand" not in label_map:
                continue
            n += _export_mi_epochs(
                {sub_id: {sess_id: {run_id: raw}}},
                out_dir=out_dir,
                dataset_prefix="liu2024",
                subject_tag="S",
                cfg=cfg,
                label_map=label_map,
            )
    print(f"Liu2024 stroke: {n} parquet exports -> {out_dir}")
    return n


def preprocess_liu2025(out_dir: Path, *, max_subjects: int = 27) -> int:
    """Longitudinal lower-limb stroke MI — Liu et al. 2025, via MOABB."""
    try:
        from moabb.datasets import Liu2025
    except ImportError:
        print("Liu2025 not in this MOABB version — pip install -U moabb")
        return 0

    from mes_core.preprocessing import PreprocessConfig, preprocess_raw

    ds = Liu2025()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PreprocessConfig()
    n = 0
    for sid in ds.subject_list[:max_subjects]:
        try:
            raw_dict = ds.get_data(subjects=[sid])
        except Exception as e:
            print(f"liu2025 skip {sid}: {e}")
            continue
        import mne

        for sub_id, sess_id, run_id, raw in _iter_moabb_raws(raw_dict):
            raw_pp = preprocess_raw(raw, cfg)
            _events, event_id = mne.events_from_annotations(raw_pp, verbose="ERROR")
            label_map: dict[str, int] = {}
            for k, code in event_id.items():
                kl = k.lower()
                if "right" in kl:
                    label_map[k] = code
                elif "rest" in kl or "break" in kl or "idle" in kl:
                    label_map["rest"] = code
            if not any("right" in k.lower() for k in label_map):
                continue
            if "rest" not in label_map:
                for k, code in event_id.items():
                    if code not in label_map.values():
                        label_map["rest"] = code
                        break
            n += _export_mi_epochs(
                {sub_id: {sess_id: {run_id: raw}}},
                out_dir=out_dir,
                dataset_prefix="liu2025",
                subject_tag="S",
                cfg=cfg,
                label_map=label_map,
            )
    print(f"Liu2025 stroke: {n} parquet exports -> {out_dir}")
    return n


def preprocess_stroke(out_dir: Path, *, max_subjects: int = 50) -> int:
    return preprocess_liu2024(out_dir, max_subjects=max_subjects) + preprocess_liu2025(
        out_dir, max_subjects=min(27, max_subjects)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/mes-out/processed"))
    ap.add_argument("--bci-iv-2b", action="store_true")
    ap.add_argument("--stroke", action="store_true")
    ap.add_argument("--max-subjects", type=int, default=50)
    args = ap.parse_args()
    total = 0
    if args.bci_iv_2b:
        total += preprocess_bci_iv_2b(args.out)
    if args.stroke:
        total += preprocess_stroke(args.out, max_subjects=args.max_subjects)
    if not args.bci_iv_2b and not args.stroke:
        total += preprocess_bci_iv_2b(args.out)
        total += preprocess_stroke(args.out, max_subjects=args.max_subjects)
    return 0 if total >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
