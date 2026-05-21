#!/usr/bin/env python3
"""CI gate: stroke benchmark metrics must not regress below committed floors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benchmarks",
        type=Path,
        default=ROOT / "mes_core/data/benchmarks.json",
    )
    ap.add_argument(
        "--gates",
        type=Path,
        default=ROOT / "mes_core/data/stroke_validation_gates.json",
    )
    ap.add_argument("--skip-if-missing", action="store_true")
    args = ap.parse_args()

    if not args.benchmarks.exists():
        if args.skip_if_missing:
            print("skip: no benchmarks.json")
            return 0
        raise SystemExit("benchmarks.json missing — run scripts/run_stroke_pipeline.py")

    bench = json.loads(args.benchmarks.read_text())
    gates = json.loads(args.gates.read_text()) if args.gates.exists() else {}
    stroke = bench.get("stroke_liu2024") or {}
    auc = float(stroke.get("mes_auc", 0))
    d = float(stroke.get("mes_cohen_d", 0))
    min_auc = float(gates.get("min_mes_auc", 0.5))
    min_d = float(gates.get("min_cohen_d", 0.2))

    print(f"stroke MES AUC={auc:.3f} (min {min_auc}) Cohen d={d:.3f} (min {min_d})")
    ok = auc >= min_auc and d >= min_d
    if not ok:
        print("FAIL: stroke metrics below regression gates")
        return 1
    print("PASS: stroke regression gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
