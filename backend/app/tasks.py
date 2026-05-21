"""Background processing pipeline: ingest -> preprocess -> features -> MES -> persist."""

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
from mes_core.preprocessing import (
    PreprocessConfig,
    epoch_raw,
    epoch_sliding_windows,
    preprocess_raw,
)
from mes_core.scoring.recovery import mes_recovery_z
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


def _prior_mes_means(participant_id: str, exclude_session_id: str) -> list[float]:
    with session_scope() as db:
        rows = (
            db.query(MesScore.mes_mean)
            .join(DbSession, MesScore.session_id == DbSession.id)
            .filter(
                DbSession.participant_id == participant_id,
                DbSession.status == "done",
                DbSession.id != exclude_session_id,
            )
            .all()
        )
    return [float(r[0]) for r in rows]


def run_session_pipeline(
    session_id: str,
    upload_path: str,
    task: str,
    *,
    cohort: str = "healthy",
    had_rest_block: bool = True,
) -> None:
    """Full pipeline entrypoint executed as a BackgroundTask."""
    log.info("session_pipeline_start", session_id=session_id, path=upload_path, cohort=cohort)
    try:
        _update(session_id, status="processing", progress=5)

        raw = load_eeg(upload_path)
        _update(session_id, progress=20)

        n_ch = len(raw.info["ch_names"])
        cfg = PreprocessConfig(do_ica=n_ch >= 32)
        raw_pp = preprocess_raw(raw, cfg)
        _update(session_id, progress=45)

        epochs = epoch_raw(raw_pp)
        data = epochs.get_data() if epochs is not None and len(epochs) > 0 else None
        if data is None or len(data) == 0:
            data = epoch_sliding_windows(raw_pp, window_s=6.0, step_s=3.0)
        if data is None or len(data) == 0:
            arr = raw_pp.get_data()
            n_ch, n_t = arr.shape
            n_win = int(6.0 * raw_pp.info["sfreq"])
            if n_t < n_win:
                pad = np.zeros((n_ch, n_win - n_t), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=1)
            data = arr[None, ...]
        _update(session_id, progress=65)

        ch_names = raw_pp.info["ch_names"]
        used_sliding = data is not None and (
            epochs is None or len(epochs) == 0
        ) and len(data) > 1

        try:
            scored = score_epochs(
                data,
                sfreq=float(raw_pp.info["sfreq"]),
                ch_names=ch_names,
                task=task,
                use_onnx=True,
                cohort=cohort if cohort in ("healthy", "stroke") else "healthy",
                require_quality=True,
                had_rest_block=had_rest_block,
                used_sliding_windows=used_sliding,
            )
        except ValueError as qe:
            log.warning("quality_gate_retry", err=str(qe))
            scored = score_epochs(
                data,
                sfreq=float(raw_pp.info["sfreq"]),
                ch_names=ch_names,
                task=task,
                use_onnx=True,
                cohort=cohort if cohort in ("healthy", "stroke") else "healthy",
                require_quality=False,
                had_rest_block=had_rest_block,
                used_sliding_windows=used_sliding,
            )
            scored.reliability = "Low"

        result = scored.mes
        model_sha = scored.model_sha
        _update(session_id, progress=80)

        from mes_core.config import BANDS

        half = data.shape[-1] // 2
        erd = erd_percent(
            data[..., half:], data[..., :half], raw_pp.info["sfreq"], BANDS["mu"]
        ).mean(axis=0)
        topo = scalp_topomap_payload(erd, ch_names)

        contra_chs, ipsi_chs = default_contra_ipsi_for_task(task)
        li_val = lateralization_index(
            erd, ch_names, contra_channels=contra_chs, ipsi_channels=ipsi_chs
        )
        if not np.isscalar(li_val):
            li_val = float(np.nanmean(li_val))
        if not np.isfinite(li_val):
            li_val = 0.0

        with session_scope() as db:
            sess = db.get(DbSession, session_id)
            pid = sess.participant_id if sess else ""
        recovery_z, recovery_label = mes_recovery_z(
            float(result.summary["mes_mean"]),
            _prior_mes_means(pid, session_id),
        )

        score_meta = {
            "quality": scored.quality,
            "baseline_kind": scored.baseline_kind,
            "cohort": scored.cohort,
            "had_rest_block": had_rest_block,
            "recovery_label": recovery_label,
            "posterior_entropy": scored.posterior_entropy,
            "n_rest_epochs": scored.n_rest_epochs,
            "n_task_epochs": scored.n_task_epochs,
        }

        report_uri = _render_and_store_report(
            session_id=session_id,
            task=task,
            result=result,
            erd_topomap=topo,
            lateralization=float(li_val),
            model_sha=model_sha or "unknown",
            reliability=scored.reliability,
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
                raw_features={
                    k: list(map(float, v.tolist())) for k, v in result.raw_features.items()
                },
                model_sha=model_sha,
                report_uri=report_uri,
                reliability=scored.reliability,
                mes_recovery_z=recovery_z,
                score_meta=score_meta,
            )
            db.add(score)
            sess = db.get(DbSession, session_id)
            if sess:
                sess.status = "done"
                sess.progress = 100
                sess.completed_at = dt.datetime.now(dt.UTC)
            db.commit()
        log.info(
            "session_pipeline_done",
            session_id=session_id,
            mes_mean=result.summary["mes_mean"],
            reliability=scored.reliability,
        )
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
    reliability: str = "Medium",
) -> str | None:
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
                model_sha=f"{model_sha} · {reliability} reliability",
                created_at=dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
                notes=participant.notes or "",
            )
        html = build_session_report_html(ctx)
        out = Path(tempfile.gettempdir()) / f"mes_report_{session_id}.pdf"
        render_pdf(html, out)
        return store_report(out, session_id)
    except Exception as e:
        log.warning("report_render_failed", session_id=session_id, err=str(e))
        return None
