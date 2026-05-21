"""Meta endpoints: health, models, benchmarks, demo bootstrap."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.db.models import MesScore, Participant
from backend.app.db.models import Session as DbSession
from backend.app.db.session import get_session
from backend.app.schemas import HealthOut, SessionOut
from mes_core import __version__
from mes_core.config import HF_REPOS, cache_root

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/healthz", response_model=HealthOut)
def healthz() -> HealthOut:
    return HealthOut(
        status="ok",
        version=__version__,
        sqlite_path=str(cache_root() / "mes.db"),
        model_repo=HF_REPOS.model,
    )


@router.get("/models")
def models() -> dict:
    info = {
        "model_repo": HF_REPOS.model,
        "dataset_repo": HF_REPOS.dataset,
        "available": [],
        "benchmarks": None,
    }
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        files = api.list_repo_files(HF_REPOS.model)
        info["available"] = [f for f in files if f.endswith(".onnx")]
        for cand in files:
            if cand == "benchmarks.json":
                from huggingface_hub import hf_hub_download
                p = hf_hub_download(repo_id=HF_REPOS.model, filename="benchmarks.json",
                                    token=os.environ.get("HF_TOKEN"))
                info["benchmarks"] = json.loads(Path(p).read_text())
                break
    except Exception as e:
        info["error"] = str(e)
    return info


@router.post("/demo/seed", response_model=list[SessionOut])
def seed_demo(db: Session = Depends(get_session)) -> list[DbSession]:
    """Create two preloaded demo sessions backed by mock MES scores.

    Used by the frontend demo mode so med-school presentations work out-of-the-box
    even before any real EEG file has been uploaded.
    """
    sessions = []
    for code, task, mes_mean, mes_std, traj in (
        (
            "HC-001",
            "right_hand",
            71.0,
            8.0,
            [55, 62, 68, 72, 75, 70, 78, 80, 76, 74, 79, 82, 77, 75, 81],
        ),
        (
            "ST-014",
            "right_hand",
            38.0,
            12.0,
            [22, 26, 30, 34, 38, 42, 40, 45, 48, 50, 47, 52, 55, 58, 62],
        ),
    ):
        p = db.query(Participant).filter_by(code=code).first()
        if p is None:
            p = Participant(code=code, notes=f"Demo participant ({code}).")
            db.add(p)
            db.flush()

        existing = db.query(DbSession).filter_by(participant_id=p.id, is_demo=1).first()
        if existing:
            sessions.append(existing)
            continue

        s = DbSession(
            participant_id=p.id, task=task, target_limb="Right hand",
            headset="OpenBCI Cyton+Daisy", montage="openbci_16",
            original_filename="demo.edf", status="done", progress=100, is_demo=1,
        )
        db.add(s)
        db.flush()
        import numpy as np

        from mes_core.config import OPENBCI_MONTAGE_16
        from mes_core.viz.topomap import scalp_topomap_payload

        rng = np.random.default_rng(hash(code) & 0xFFFF)
        erd_vals = rng.normal(mes_mean / 2.0, 10, len(OPENBCI_MONTAGE_16))
        topo = scalp_topomap_payload(erd_vals, list(OPENBCI_MONTAGE_16))
        score = MesScore(
            session_id=s.id,
            mes_mean=float(mes_mean),
            mes_median=float(np.median(traj)),
            mes_std=float(mes_std),
            n_trials=len(traj),
            lateralization=0.42 if "HC" in code else 0.12,
            mes_per_trial=[float(v) for v in traj],
            erd_topomap=topo,
            raw_features={
                "z_mu": rng.normal(1.2 if "HC" in code else 0.3, 0.4, len(traj)).tolist(),
                "z_beta": rng.normal(0.8 if "HC" in code else 0.2, 0.4, len(traj)).tolist(),
                "z_li": rng.normal(0.5 if "HC" in code else 0.1, 0.3, len(traj)).tolist(),
                "z_mrcp": rng.normal(-0.5 if "HC" in code else -0.1, 0.5, len(traj)).tolist(),
            },
            model_sha="demo",
        )
        db.add(score)
        sessions.append(s)
    db.commit()
    for s in sessions:
        db.refresh(s)
    return sessions
