#!/usr/bin/env python3
"""Autonomous protocol E2E: generate file, score, optional live API upload."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _score_local(path: Path, had_rest_block: bool) -> dict:
    from mes_core.pipeline import score_recording

    r = score_recording(path, had_rest_block=had_rest_block, require_quality=False)
    d = r.to_dict()
    return {
        "had_rest_block": had_rest_block,
        "reliability": r.reliability,
        "baseline_kind": r.baseline_kind,
        "n_rest_epochs": r.n_rest_epochs,
        "n_task_epochs": r.n_task_epochs,
        "mes_mean": d.get("mes_mean") or r.mes.summary.get("mes_mean"),
        "mes_median": d.get("mes_median") or r.mes.summary.get("mes_median"),
        "model_sha": r.model_sha,
        "posterior_entropy": r.posterior_entropy,
    }


def _api_upload(base_url: str, path: Path, had_rest_block: bool) -> dict:
    import httpx

    base = base_url.rstrip("/")
    with httpx.Client(base_url=base, timeout=120.0) as client:
        hz = client.get("/api/healthz")
        hz.raise_for_status()
        p = client.post("/api/participants", json={"code": f"E2E-{int(time.time())}"})
        if p.status_code == 409:
            p = client.post("/api/participants", json={"code": f"E2E-{int(time.time())}-2"})
        p.raise_for_status()
        pid = p.json()["id"]
        with path.open("rb") as f:
            r = client.post(
                "/api/sessions",
                files={"file": (path.name, f, "text/plain")},
                data={
                    "participant_id": pid,
                    "task": "right_hand",
                    "target_limb": "Right hand",
                    "headset": "OpenBCI Cyton+Daisy",
                    "montage": "openbci_16",
                    "cohort": "healthy",
                    "had_rest_block": "true" if had_rest_block else "false",
                },
            )
        r.raise_for_status()
        sid = r.json()["id"]
        for _ in range(120):
            st = client.get(f"/api/sessions/{sid}").json()
            if st["status"] in ("done", "failed"):
                break
            time.sleep(1.0)
        if st["status"] != "done":
            return {"error": "pipeline_failed", "session": st}
        score = client.get(f"/api/sessions/{sid}/score").json()
        return {"session_id": sid, "status": st["status"], "score": score}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "data/protocol_test_openbci.txt")
    ap.add_argument("--live", type=str, default="", help="Base URL e.g. https://abachu2005-mes.hf.space")
    ap.add_argument("--json-out", type=Path, default="")
    args = ap.parse_args()

    from scripts.generate_protocol_test_file import build_protocol_recording, write_openbci_txt

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = build_protocol_recording()
    write_openbci_txt(args.output, data)

    report: dict = {"file": str(args.output), "local": {}}
    report["local"]["without_rest"] = _score_local(args.output, False)
    report["local"]["with_rest"] = _score_local(args.output, True)

    if args.live:
        try:
            report["live"] = _api_upload(args.live, args.output, had_rest_block=True)
        except Exception as e:
            report["live"] = {"error": str(e)}

    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n", encoding="utf-8")

    wr = report["local"]["with_rest"]
    ok = wr["n_rest_epochs"] >= 10 and wr["baseline_kind"] == "subject_rest"
    if args.live and "score" in report.get("live", {}):
        sc = report["live"]["score"]
        ok = ok and sc.get("reliability") in ("High", "Medium", "Low")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
