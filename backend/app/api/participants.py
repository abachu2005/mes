"""Participant endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.db.models import MesScore, Participant, Session as DbSession
from backend.app.db.session import get_session
from backend.app.schemas import (
    LongitudinalPoint,
    ParticipantCreate,
    ParticipantLongitudinal,
    ParticipantOut,
)

router = APIRouter(prefix="/api/participants", tags=["participants"])


@router.get("", response_model=list[ParticipantOut])
def list_participants(db: Session = Depends(get_session)) -> list[Participant]:
    return list(db.query(Participant).order_by(Participant.created_at.desc()).all())


@router.post("", response_model=ParticipantOut, status_code=201)
def create_participant(body: ParticipantCreate, db: Session = Depends(get_session)) -> Participant:
    existing = db.query(Participant).filter_by(code=body.code).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"participant code {body.code!r} already exists")
    p = Participant(code=body.code, notes=body.notes or "")
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@router.get("/{participant_id}", response_model=ParticipantOut)
def get_participant(participant_id: str, db: Session = Depends(get_session)) -> Participant:
    p = db.get(Participant, participant_id)
    if p is None:
        raise HTTPException(status_code=404, detail="participant not found")
    return p


@router.get("/{participant_id}/longitudinal", response_model=ParticipantLongitudinal)
def get_longitudinal(participant_id: str, db: Session = Depends(get_session)) -> ParticipantLongitudinal:
    p = db.get(Participant, participant_id)
    if p is None:
        raise HTTPException(status_code=404, detail="participant not found")
    sessions = (
        db.query(DbSession)
        .filter_by(participant_id=participant_id)
        .order_by(DbSession.created_at.asc())
        .all()
    )
    points = []
    for s in sessions:
        score: MesScore | None = s.score
        points.append(
            LongitudinalPoint(
                session_id=s.id,
                created_at=s.created_at,
                task=s.task,
                mes_mean=float(score.mes_mean) if score else None,
                mes_std=float(score.mes_std) if score else None,
                status=s.status,
            )
        )
    return ParticipantLongitudinal(
        participant=ParticipantOut.model_validate(p),
        points=points,
    )
