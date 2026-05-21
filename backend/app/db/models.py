"""SQLAlchemy ORM models for participants, sessions, and MES scores."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Participant(Base):
    __tablename__ = "participants"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    notes: Mapped[str | None] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)

    sessions: Mapped[list[Session]] = relationship(
        "Session", back_populates="participant", cascade="all, delete-orphan"
    )


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    participant_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("participants.id", ondelete="CASCADE"), index=True
    )
    task: Mapped[str] = mapped_column(String(64))
    target_limb: Mapped[str] = mapped_column(String(64))
    headset: Mapped[str] = mapped_column(String(64), default="OpenBCI Cyton+Daisy")
    montage: Mapped[str] = mapped_column(String(64), default="openbci_16")
    original_filename: Mapped[str | None] = mapped_column(String(255), default=None)
    file_uri: Mapped[str | None] = mapped_column(String(512), default=None)
    status: Mapped[str] = mapped_column(String(32), default="queued")  # queued/processing/done/failed
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, default=None)
    is_demo: Mapped[int] = mapped_column(Integer, default=0)

    participant: Mapped[Participant] = relationship("Participant", back_populates="sessions")
    score: Mapped[MesScore] = relationship(
        "MesScore", back_populates="session", uselist=False, cascade="all, delete-orphan"
    )


class MesScore(Base):
    __tablename__ = "mes_scores"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("sessions.id", ondelete="CASCADE"), unique=True, index=True
    )
    mes_mean: Mapped[float] = mapped_column(Float)
    mes_median: Mapped[float] = mapped_column(Float)
    mes_std: Mapped[float] = mapped_column(Float)
    n_trials: Mapped[int] = mapped_column(Integer)
    lateralization: Mapped[float] = mapped_column(Float)
    mes_per_trial: Mapped[list] = mapped_column(JSON)
    erd_topomap: Mapped[dict] = mapped_column(JSON)
    raw_features: Mapped[dict] = mapped_column(JSON)
    model_sha: Mapped[str | None] = mapped_column(String(64), default=None)
    report_uri: Mapped[str | None] = mapped_column(String(512), default=None)
    reliability: Mapped[str | None] = mapped_column(String(16), default="Medium")
    mes_recovery_z: Mapped[float | None] = mapped_column(Float, default=None)
    score_meta: Mapped[dict | None] = mapped_column(JSON, default=None)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)

    session: Mapped[Session] = relationship("Session", back_populates="score")
