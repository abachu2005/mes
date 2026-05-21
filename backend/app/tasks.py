"""Background processing pipeline: ingest -> preprocess -> features -> MES -> persist.

Executed via FastAPI's BackgroundTasks. Idempotent on session_id.
"""

from __future__ import annotations

import datetime as dt
import tempfile
import traceback
from pathlib import Path

import numpy as np
import structlog

from backend.app.db.models import MesScore
from backend.app.db.models import Session as DbSession
from backend.app.db.session import session_scope
from backend.app.storage import store_report
from mes_core.features.bandpower import erd_percent
from mes_core.features.lateralization import default_contra_ipsi_for_task, lateralization_index
from mes_core.io import load_eeg
from mes_core.pipeline import score_epochs
from mes_core.preprocessing import PreprocessConfig, epoch_raw, preprocess_raw
from mes_core.viz.report import ReportContext, build_session_report_html, render_pdf
from mes_core.viz.topomap import scalp_topomap_payload

log = structlog.get_logger(__name__)


def _update(session_id: str, **fields: object) -> None:
    db = session_scope()
    try:
        sess = db.get(DbSession, session_id)
        if sess is None:
            return
        for k, v in fields.items():
            setattr(sess, k, v)
        db.commit()
    finally:
        db.close()


def run_session_pipeline(session_id: str, upload_path: str, task: str) -> None:
    """Full pipeline entrypoint executed as a BackgroundTask."""
    log.info("session_pipeline_start", session_id=session_id, path=upload_path)
    try:
        _update(session_id, status="processing", progress=5)

        raw = load_eeg(upload_path)
        _update(session_id, progress=20)

        cfg = PreprocessConfig(do_ica=True)
        raw_pp = preprocess_raw(raw, cfg)
        _update(session_id, progress=45)

        epochs = epoch_raw(raw_pp)
        data = epochs.get_data() if epochs is not None and len(epochs) > 0 else None
        if data is None or len(data) == 0:
            # Fall back to whole-recording windowed epochs so we always score something.
            arr = raw_pp.get_data()
            n_ch, n_t = arr.shape
            n_win = int(6.0 * raw_pp.info["sfreq"])
            if n_t < n_win:
                pad = np.zeros((n_ch, n_win - n_t), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=1)
            data = arr[None, ...]
        _update(session_id, progress=65)

        ch_names = raw_pp.info["ch_names"]
        scored = score_epochs(
            data,
            sfreq=float(raw_pp.info["sfreq"]),
            ch_names=ch_names,
            task=task,
            use_onnx=True,
        )
        result = scored.mes
        model_sha = scored.model_sha
        _update(session_id, progress=80)

        # ERD topomap from feature contra
        from mes_core.config import BANDS

        half = data.shape[-1] // 2
        erd = erd_percent(data[..., half:], data[..., :half], raw_pp.info["sfreq"], BANDS["mu"]).mean(axis=0)
        topo = scalp_topomap_payload(erd, ch_names)

        contra_chs, ipsi_chs = default_contra_ipsi_for_task(task)
        li_val = lateralization_index(
            erd, ch_names, contra_channels=contra_chs, ipsi_channels=ipsi_chs
        )
        if not np.isscalar(li_val):
            li_val = float(np.nanmean(li_val))
        if not np.isfinite(li_val):
            li_val = 0.0

        report_uri = _render_and_store_report(
            session_id=session_id,
            task=task,
            result=result,
            erd_topomap=topo,
            lateralization=float(li_val),
            model_sha=model_sha or "unknown",
        )

        with session_scope() as db:
            score = MesScore(
                session_id=session_id,
                mes_mean=float(result.summary["mes_mean"]),
                mes_median=float(result.summary["mes_median"]),
                mes_std=float(result.summary["mes_std"]),
                n_trials=int(result.summary["n_trials"]),
                lateralization=float(li_val),
                mes_per_trial=[float(v) for v in result.mes_per_trial.tolist()],
                erd_topomap=topo,
                raw_features={k: list(map(float, v.tolist())) for k, v in result.raw_features.items()},
                model_sha=model_sha,
                report_uri=report_uri,
            )
            db.add(score)
            sess = db.get(DbSession, session_id)
            if sess:
                sess.status = "done"
                sess.progress = 100
                sess.completed_at = dt.datetime.now(dt.UTC)
            db.commit()
        log.info("session_pipeline_done", session_id=session_id, mes_mean=result.summary["mes_mean"])
    except Exception as e:
        log.exception("session_pipeline_failed", session_id=session_id)
        _update(
            session_id,
            status="failed",
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}",
        )


def _render_and_store_report(
    session_id: str,
    task: str,
    result,
    erd_topomap: dict,
    lateralization: float,
    model_sha: str,
) -> str | None:
    """Render PDF report and upload to HF Hub (best effort)."""
    try:
        with session_scope() as db:
            sess = db.get(DbSession, session_id)
            if sess is None:
                return None
            participant = sess.participant
            ctx = ReportContext(
                participant_code=participant.code,
                session_id=session_id,
                task=task,
                target_limb=sess.target_limb,
                headset=sess.headset,
                mes_mean=float(result.summary["mes_mean"]),
                mes_median=float(result.summary["mes_median"]),
                mes_std=float(result.summary["mes_std"]),
                n_trials=int(result.summary["n_trials"]),
                mes_per_trial=[float(v) for v in result.mes_per_trial.tolist()],
                erd_topomap=erd_topomap,
                lateralization=lateralization,
                model_sha=model_sha,
                created_at=dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
                notes=participant.notes or "",
            )
        html = build_session_report_html(ctx)
        out = Path(tempfile.gettempdir()) / f"mes_report_{session_id}.pdf"
        render_pdf(html, out)
        uri = store_report(out, session_id)
        return uri
    except Exception as e:
        log.warning("report_render_failed", session_id=session_id, err=str(e))
        return None
