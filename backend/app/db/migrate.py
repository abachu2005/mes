"""Lightweight SQLite migrations for additive columns."""

from __future__ import annotations

from sqlalchemy import inspect, text

from backend.app.db.session import _engine


def migrate_sqlite() -> None:
    if not str(_engine.url).startswith("sqlite"):
        return
    insp = inspect(_engine)
    if "mes_scores" not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns("mes_scores")}
    additions = {
        "reliability": "VARCHAR(16)",
        "mes_recovery_z": "FLOAT",
        "score_meta": "JSON",
    }
    with _engine.begin() as conn:
        for name, typ in additions.items():
            if name not in cols:
                conn.execute(text(f"ALTER TABLE mes_scores ADD COLUMN {name} {typ}"))
