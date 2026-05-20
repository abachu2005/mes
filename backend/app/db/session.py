"""Database session + engine factories."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session as ORMSession
from sqlalchemy.orm import sessionmaker

from backend.app.db.models import Base


def _default_db_url() -> str:
    base = os.environ.get("MES_DATA") or os.path.expanduser("~/.cache/mes")
    Path(base).mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{base}/mes.db"


DB_URL = os.environ.get("DATABASE_URL", _default_db_url())

_engine = create_engine(
    DB_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {},
)
_SessionFactory = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(_engine)


def get_session() -> Generator[ORMSession, None, None]:
    db = _SessionFactory()
    try:
        yield db
    finally:
        db.close()


def session_scope() -> ORMSession:
    """Return a new session - caller must close."""
    return _SessionFactory()
