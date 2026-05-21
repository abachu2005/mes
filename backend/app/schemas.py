"""Pydantic schemas for the HTTP API."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParticipantBase(BaseModel):
    code: str = Field(..., min_length=1, max_length=64)
    notes: str | None = ""


class ParticipantCreate(ParticipantBase):
    pass


class ParticipantOut(ParticipantBase):
    model_config = ConfigDict(from_attributes=True)
    id: str
    created_at: dt.datetime


class SessionCreate(BaseModel):
    participant_id: str
    task: str = "right_hand"
    target_limb: str = "Right hand"
    headset: str = "OpenBCI Cyton+Daisy"
    montage: str = "openbci_16"
    cohort: str = "healthy"  # healthy | stroke
    had_rest_block: bool = True


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    participant_id: str
    task: str
    target_limb: str
    headset: str
    montage: str
    original_filename: str | None
    status: str
    progress: int
    error: str | None
    is_demo: int
    created_at: dt.datetime
    completed_at: dt.datetime | None


class ScoreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    session_id: str
    mes_mean: float
    mes_median: float
    mes_std: float
    n_trials: int
    lateralization: float
    mes_per_trial: list[float]
    erd_topomap: dict[str, Any]
    raw_features: dict[str, Any]
    model_sha: str | None
    reliability: str | None = None
    mes_recovery_z: float | None = None
    score_meta: dict[str, Any] | None = None
    created_at: dt.datetime


class LongitudinalPoint(BaseModel):
    session_id: str
    created_at: dt.datetime
    task: str
    mes_mean: float | None
    mes_std: float | None
    status: str


class ParticipantLongitudinal(BaseModel):
    participant: ParticipantOut
    points: list[LongitudinalPoint]


class HealthOut(BaseModel):
    status: str
    version: str
    sqlite_path: str
    model_repo: str
