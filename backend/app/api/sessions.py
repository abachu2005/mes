"""Session endpoints: upload, status, score, report."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.db.models import Participant
from backend.app.db.models import Session as DbSession
from backend.app.db.session import get_session
from backend.app.schemas import ScoreOut, SessionOut
from backend.app.storage import get_local_path, store_upload
from backend.app.tasks import run_session_pipeline

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionOut, status_code=201)
async def upload_session(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    participant_id: str = Form(...),
    task: str = Form("right_hand"),
    target_limb: str = Form("Right hand"),
    headset: str = Form("OpenBCI Cyton+Daisy"),
    montage: str = Form("openbci_16"),
    cohort: str = Form("healthy"),
    had_rest_block: str = Form("true"),
    db: Session = Depends(get_session),
) -> DbSession:
    p = db.get(Participant, participant_id)
    if p is None:
        raise HTTPException(status_code=404, detail="participant not found")

    suffix = Path(file.filename or "upload.edf").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    sess = DbSession(
        participant_id=participant_id,
        task=task,
        target_limb=target_limb,
        headset=headset,
        montage=montage,
        original_filename=file.filename,
        status="queued",
        progress=1,
    )
    db.add(sess)
    db.commit()
    db.refresh(sess)
    sess.file_uri = store_upload(tmp_path, sess.id, original_filename=file.filename)
    db.commit()
    db.refresh(sess)
    background.add_task(
        run_session_pipeline,
        sess.id,
        tmp_path,
        task,
        cohort=cohort if cohort in ("healthy", "stroke") else "healthy",
        had_rest_block=had_rest_block.lower() in ("true", "1", "yes"),
    )
    return sess


@router.get("", response_model=list[SessionOut])
def list_sessions(
    participant_id: str | None = None,
    db: Session = Depends(get_session),
) -> list[DbSession]:
    q = db.query(DbSession).order_by(DbSession.created_at.desc())
    if participant_id:
        q = q.filter_by(participant_id=participant_id)
    return list(q.all())


@router.get("/{session_id}", response_model=SessionOut)
def get_session_status(session_id: str, db: Session = Depends(get_session)) -> DbSession:
    s = db.get(DbSession, session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    return s


@router.get("/{session_id}/score", response_model=ScoreOut)
def get_score(session_id: str, db: Session = Depends(get_session)) -> ScoreOut:
    s = db.get(DbSession, session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    if s.score is None:
        raise HTTPException(status_code=409, detail=f"session is {s.status}")
    return ScoreOut.model_validate(s.score)


@router.get("/{session_id}/report.pdf")
def get_report(session_id: str, db: Session = Depends(get_session)) -> FileResponse:
    s = db.get(DbSession, session_id)
    if s is None or s.score is None:
        raise HTTPException(status_code=404, detail="report not available")
    if s.score.report_uri is None:
        raise HTTPException(status_code=404, detail="report not generated")
    local = get_local_path(s.score.report_uri)
    if local is None or not local.exists():
        raise HTTPException(status_code=404, detail="report file missing")
    return FileResponse(local, media_type="application/pdf", filename=f"mes_{session_id}.pdf")


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str, db: Session = Depends(get_session)) -> None:
    s = db.get(DbSession, session_id)
    if s is None:
        return
    db.delete(s)
    db.commit()
