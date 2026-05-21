"""End-to-end API integration tests using FastAPI's TestClient."""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    db_dir = tmp_path_factory.mktemp("dbroot")
    os.environ["MES_DATA"] = str(db_dir)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_dir}/test.db"
    # Reload module so the engine uses the new URL.
    import importlib

    from backend.app.db import session as session_mod
    importlib.reload(session_mod)
    from backend.app import main as main_mod
    importlib.reload(main_mod)
    return TestClient(main_mod.app)


def test_healthz(client: TestClient) -> None:
    r = client.get("/api/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_create_and_list_participants(client: TestClient) -> None:
    r = client.post("/api/participants", json={"code": "T001", "notes": "test"})
    assert r.status_code == 201, r.text
    pid = r.json()["id"]

    r = client.get("/api/participants")
    assert r.status_code == 200
    assert any(p["id"] == pid for p in r.json())

    # Duplicate code -> 409
    r = client.post("/api/participants", json={"code": "T001"})
    assert r.status_code == 409


def test_demo_seed_creates_two_sessions(client: TestClient) -> None:
    r = client.post("/api/demo/seed")
    assert r.status_code == 200, r.text
    sessions = r.json()
    assert len(sessions) == 2
    # Idempotent
    r2 = client.post("/api/demo/seed")
    assert r2.status_code == 200
    assert len(r2.json()) == 2


def test_session_upload_and_pipeline_round_trip(client: TestClient, tmp_path) -> None:
    # Create participant
    p = client.post("/api/participants", json={"code": "P-RT"}).json()

    # Generate a small synthetic EDF on disk.
    import mne
    import numpy as np

    from mes_core.config import OPENBCI_MONTAGE_16

    sfreq = 125.0
    n = int(20 * sfreq)
    data = np.random.default_rng(0).standard_normal((16, n)) * 1e-6
    info = mne.create_info(list(OPENBCI_MONTAGE_16), sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    edf_path = tmp_path / "rt.edf"
    mne.export.export_raw(str(edf_path), raw, fmt="edf", overwrite=True, verbose="ERROR")

    with edf_path.open("rb") as f:
        r = client.post(
            "/api/sessions",
            files={"file": ("rt.edf", f, "application/octet-stream")},
            data={
                "participant_id": p["id"],
                "task": "right_hand",
                "target_limb": "Right hand",
                "headset": "OpenBCI Cyton+Daisy",
                "montage": "openbci_16",
            },
        )
    assert r.status_code == 201, r.text
    sid = r.json()["id"]

    # BackgroundTasks in TestClient execute synchronously after the response.
    # Poll a few times to be safe.
    for _ in range(60):
        st = client.get(f"/api/sessions/{sid}").json()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    final = client.get(f"/api/sessions/{sid}").json()
    assert final["status"] == "done", final
    assert final["progress"] == 100

    score = client.get(f"/api/sessions/{sid}/score").json()
    assert 0 <= score["mes_mean"] <= 100
    assert score["n_trials"] >= 1
    assert "erd_topomap" in score
    model_sha = score.get("model_sha") or ""
    assert model_sha not in ("", "heuristic"), (
        f"Expected ONNX inference, got model_sha={model_sha!r}"
    )
    assert "eegnet" in model_sha.lower() or "ensemble" in model_sha.lower(), (
        f"Expected EEGNet or ensemble in model_sha, got {model_sha!r}"
    )
