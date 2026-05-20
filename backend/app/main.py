"""MES FastAPI application entry point."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api import meta, participants, sessions
from backend.app.db.session import init_db
from mes_core import __version__

STATIC_DIR = Path(__file__).parent / "static"


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(message)s",
        stream=sys.stdout,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def create_app() -> FastAPI:
    _configure_logging()
    init_db()
    app = FastAPI(
        title="MES — Motor Engagement Signal",
        version=__version__,
        description="Quantifying neural drive for movement recovery from EEG.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # CORS only matters when frontend is served on a different origin (local dev).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(meta.router)
    app.include_router(participants.router)
    app.include_router(sessions.router)

    # Serve the built React frontend if present (in production).
    if STATIC_DIR.exists():
        # Mount /assets so Vite chunks resolve.
        if (STATIC_DIR / "assets").exists():
            app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")
        index = STATIC_DIR / "index.html"

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str) -> FileResponse:
            target = STATIC_DIR / full_path
            if target.is_file():
                return FileResponse(target)
            return FileResponse(index)

    return app


app = create_app()


def main() -> None:
    """Used by `python -m backend.app.main` for local runs."""
    import uvicorn

    if "--smoke" in sys.argv:
        # Don't actually serve; just verify the app object builds and DB initializes.
        print("smoke ok:", app.title, app.version)
        return
    uvicorn.run("backend.app.main:app", host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "7860")), reload=False)


if __name__ == "__main__":
    main()
