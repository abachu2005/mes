"""ONNX inference wrapper used by the backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)

_BUNDLED_MODELS = Path(__file__).resolve().parents[1] / "data" / "models"
_RIEMANN_FRONTEND_CACHE: dict[str, tuple[Any, Any]] = {}


@dataclass
class OnnxClassifier:
    """Tiny inference wrapper around an ONNX classifier model.

    Expects a model whose:
      - input is named like 'X' or 'input' with shape (batch, ...)
      - first output is probabilities of shape (batch, n_classes)
    """

    session: Any
    input_name: str
    output_name: str
    n_classes: int
    feature_shape: tuple[int, ...]
    metadata: dict[str, Any]

    @classmethod
    def from_path(cls, model_path: str | Path) -> OnnxClassifier:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        in_meta = sess.get_inputs()[0]
        outputs = sess.get_outputs()
        out_meta = outputs[0]
        for o in outputs:
            if "prob" in o.name.lower():
                out_meta = o
                break
        meta = {}
        meta_path = Path(model_path).with_suffix(".json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        n_classes = (
            int(out_meta.shape[-1])
            if isinstance(out_meta.shape[-1], int)
            else int(meta.get("n_classes", 2))
        )
        shape_raw = tuple(int(d) if isinstance(d, int) else -1 for d in in_meta.shape[1:])
        return cls(
            session=sess,
            input_name=in_meta.name,
            output_name=out_meta.name,
            n_classes=n_classes,
            feature_shape=shape_raw,
            metadata=meta,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        out = self.session.run([self.output_name], {self.input_name: X})[0]
        out = np.asarray(out)
        if out.ndim == 2 and (out.min() < -1e-3 or out.max() > 1 + 1e-3):
            e = np.exp(out - out.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        return out


def _resolve_model_path(filename: str) -> Path:
    """HF Hub download with bundled-repo and cwd fallbacks."""
    from mes_core.io.hf import download_model

    bundled = _BUNDLED_MODELS / filename
    if bundled.exists():
        return bundled
    try:
        return download_model(filename)
    except Exception as e:
        log.warning("model_download_failed_falling_back_to_local", filename=filename, err=str(e))
        for candidate in (Path(filename), bundled):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Model {filename!r} not found on HF Hub or in {_BUNDLED_MODELS}"
        ) from e


def load_onnx_model(filename: str) -> tuple[OnnxClassifier, Path]:
    """Resolve a model by filename from HF Hub, bundled data, or local cache."""
    path = _resolve_model_path(filename)
    clf = OnnxClassifier.from_path(path)
    meta = dict(clf.metadata)
    meta["filename"] = filename
    return OnnxClassifier(
        session=clf.session,
        input_name=clf.input_name,
        output_name=clf.output_name,
        n_classes=clf.n_classes,
        feature_shape=clf.feature_shape,
        metadata=meta,
    ), path


# Production ONNX filenames keyed by task slug and cohort.
TASK_ONNX: dict[str, dict[str, dict[str, str]]] = {
    "right_hand": {
        "healthy": {
            "riemannian": "riemannian_lr_right_hand.onnx",
            "eegnet": "eegnet_right_hand.onnx",
        },
        "stroke": {
            "riemannian": "riemannian_lr_right_hand_stroke.onnx",
            "eegnet": "eegnet_right_hand_stroke.onnx",
        },
    },
}

# When stroke ONNX is missing, use neutral posterior so MES relies on ERD features.
STROKE_FEATURE_FIRST = True


def _task_models(task: str, cohort: str = "healthy") -> dict[str, str]:
    key = "right_hand" if "right" in task else task
    by_cohort = TASK_ONNX.get(key, TASK_ONNX["right_hand"])
    return by_cohort.get(cohort, by_cohort["healthy"])


def _is_tangent_space(clf: OnnxClassifier) -> bool:
    """Riemannian ONNX expects (batch, n_features); EEGNet expects (batch, ch, time)."""
    return len(clf.feature_shape) == 1 and clf.feature_shape[0] not in (-1, 0)


def _frontend_npz_path(onnx_path: Path) -> Path | None:
    """Sidecar tangent-space frontend saved next to the ONNX file."""
    name = onnx_path.name
    if "stroke" in name:
        candidate = onnx_path.parent / "riemannian_lr_frontend_stroke.npz"
    else:
        candidate = onnx_path.parent / "riemannian_lr_frontend.npz"
    return candidate if candidate.exists() else None


def _load_riemann_frontend(npz_path: Path) -> tuple[Any, Any]:
    key = str(npz_path.resolve())
    if key in _RIEMANN_FRONTEND_CACHE:
        return _RIEMANN_FRONTEND_CACHE[key]
    pack = np.load(npz_path, allow_pickle=True)
    cov_est = pack["cov_estimator"].item()
    ref = pack["ts_reference"]
    _RIEMANN_FRONTEND_CACHE[key] = (cov_est, ref)
    return cov_est, ref


def _tangent_space_features(epoch_data: np.ndarray, *, frontend_npz: Path) -> np.ndarray:
    """Project epochs with the training-frozen OAS + Riemann tangent space."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    cov_est, ref = _load_riemann_frontend(frontend_npz)
    cov = Covariances(estimator=cov_est)
    Xcov = cov.transform(epoch_data.astype(float))
    ts = TangentSpace(metric="riemann")
    ts.reference_ = ref
    return ts.transform(Xcov).astype(np.float32)


def _predict_target_proba(
    clf: OnnxClassifier,
    epoch_data: np.ndarray,
    *,
    model_path: Path | None = None,
) -> np.ndarray:
    """Return per-trial P(target class) for one ONNX classifier."""
    if _is_tangent_space(clf):
        onnx_path = model_path
        if onnx_path is None:
            onnx_path = _resolve_model_path(
                clf.metadata.get("filename", "riemannian_lr_right_hand.onnx")
            )
        frontend = _frontend_npz_path(onnx_path)
        if frontend is not None:
            feed = _tangent_space_features(epoch_data, frontend_npz=frontend)
        else:
            from pyriemann.estimation import Covariances
            from pyriemann.tangentspace import TangentSpace

            log.warning("riemann_frontend_missing_fit_transform_fallback", path=str(onnx_path))
            cov = Covariances(estimator="oas").fit_transform(epoch_data.astype(float))
            ts = TangentSpace(metric="riemann").fit_transform(cov)
            feed = ts.astype(np.float32)
    else:
        feed = epoch_data.astype(np.float32)
    p = clf.predict_proba(feed)
    return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()


def _fit_epoch_window(epoch_data: np.ndarray, n_times: int = 750) -> np.ndarray:
    """Crop or pad the time axis so ONNX models see the training window length."""
    x = np.asarray(epoch_data, dtype=np.float32)
    n = x.shape[-1]
    if n >= n_times:
        return x[..., :n_times]
    pad = np.zeros(x.shape[:-1] + (n_times - n,), dtype=x.dtype)
    return np.concatenate([x, pad], axis=-1)


def resolve_session_posterior(
    epoch_data: np.ndarray,
    task: str,
    *,
    cohort: str = "healthy",
    feature_first: bool | None = None,
) -> tuple[np.ndarray, str]:
    """Load ONNX classifiers for *task* and combine into p_model.

    Healthy cohort: mean of Riemannian + EEGNet when both exist.
    Stroke cohort: stroke-specific ONNX; if none load and *feature_first* is true
    (default), return 0.5 so MES weights lean on ERD/CSP features (validated path).
    Legacy shrink toward 0.5 only applies when healthy models are used for stroke.
    """
    epoch_data = _fit_epoch_window(np.asarray(epoch_data))
    n_epochs = epoch_data.shape[0]
    specs = _task_models(task, cohort)
    use_feature_first = (
        STROKE_FEATURE_FIRST if feature_first is None else feature_first
    ) and cohort == "stroke"

    posteriors: list[np.ndarray] = []
    tags: list[str] = []

    for name, filename in specs.items():
        try:
            clf, model_path = load_onnx_model(filename)
            posteriors.append(_predict_target_proba(clf, epoch_data, model_path=model_path))
            sha = clf.metadata.get("sha") or clf.metadata.get("model", name)
            tags.append(f"{name}:{sha}")
            log.info("classifier_loaded", name=name, file=filename, cohort=cohort)
        except Exception as e:
            log.warning("classifier_unavailable", name=name, file=filename, err=str(e))

    if not posteriors:
        if use_feature_first:
            return np.full(n_epochs, 0.5), "stroke_feature_first"
        raise FileNotFoundError(f"No ONNX classifiers available for task {task!r} cohort={cohort}")

    combined = np.mean(np.stack(posteriors, axis=0), axis=0)

    # Healthy models used for stroke (no stroke bundle): dampen overconfident transfer.
    if cohort == "stroke" and any("stroke" not in t for t in tags) and not use_feature_first:
        combined = np.clip(0.25 * combined + 0.75 * 0.5, 0.05, 0.95)
        tags.append("stroke_shrink")

    model_id = "ensemble(" + "+".join(tags) + ")" if len(posteriors) > 1 else tags[0]
    return combined, model_id
