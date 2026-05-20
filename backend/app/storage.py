"""Object storage adapter (HF Hub, with local-disk fallback)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import structlog

from mes_core.config import cache_root

log = structlog.get_logger(__name__)


def _local_uploads_dir() -> Path:
    p = cache_root() / "uploads"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _local_reports_dir() -> Path:
    p = cache_root() / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def store_upload(src_path: str | Path, session_id: str, *, original_filename: str | None = None) -> str:
    """Copy uploaded file to persistent storage. Returns a URI (file://... or hf://...).

    If `HF_TOKEN` is set we mirror to the sessions repo (best effort).
    """
    src = Path(src_path)
    fname = original_filename or src.name
    dest = _local_uploads_dir() / f"{session_id}__{fname}"
    shutil.copyfile(src, dest)
    uri = f"file://{dest}"

    if os.environ.get("HF_TOKEN") and os.environ.get("MES_PUBLISH_SESSIONS", "0") == "1":
        try:
            from mes_core.io.hf import upload_session_files
            upload_session_files([dest], repo_subdir=f"raw/{session_id}",
                                  commit_message=f"upload {session_id}")
            uri = f"hf://{session_id}/{dest.name}"
        except Exception as e:
            log.warning("hf_session_upload_failed", err=str(e))

    return uri


def store_report(src_path: str | Path, session_id: str) -> str:
    """Persist a generated PDF report. Returns a URI."""
    src = Path(src_path)
    dest = _local_reports_dir() / f"{session_id}.pdf"
    shutil.copyfile(src, dest)
    return f"file://{dest}"


def get_local_path(uri: str) -> Path | None:
    if uri.startswith("file://"):
        return Path(uri[7:])
    return None
