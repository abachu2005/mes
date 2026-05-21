"""ONNX inference wrapper used by the backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)


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
    def from_path(cls, model_path: str | Path) -> "OnnxClassifier":
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        in_meta = sess.get_inputs()[0]
        out_meta = sess.get_outputs()[0]
        # Try to read sidecar metadata json next to the model file.
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
        # Some models output logits - normalize with softmax if values are unbounded.
        if out.ndim == 2 and (out.min() < -1e-3 or out.max() > 1 + 1e-3):
            e = np.exp(out - out.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        return out


def load_onnx_model(filename: str) -> OnnxClassifier:
    """Resolve a model by filename from HF Hub or local cache."""
    from mes_core.io.hf import download_model

    try:
        p = download_model(filename)
    except Exception as e:
        log.warning("model_download_failed_falling_back_to_local", filename=filename, err=str(e))
        p = Path(filename)
        if not p.exists():
            raise FileNotFoundError(f"Model {filename!r} not found locally or on HF Hub")
    return OnnxClassifier.from_path(p)


# Production ONNX filenames keyed by task slug.
TASK_ONNX: dict[str, dict[str, str]] = {
    "right_hand": {
        "riemannian": "riemannian_lr_right_hand.onnx",
        "eegnet": "eegnet_right_hand.onnx",
    },
}


def _task_models(task: str) -> dict[str, str]:
    if "right" in task:
        return TASK_ONNX["right_hand"]
    return TASK_ONNX.get(task, TASK_ONNX["right_hand"])


def _is_tangent_space(clf: OnnxClassifier) -> bool:
    """Riemannian ONNX expects (batch, n_features); EEGNet expects (batch, ch, time)."""
    return len(clf.feature_shape) == 1 and clf.feature_shape[0] not in (-1, 0)


def _predict_target_proba(clf: OnnxClassifier, epoch_data: np.ndarray) -> np.ndarray:
    """Return per-trial P(target class) for one ONNX classifier."""
    if _is_tangent_space(clf):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace

        cov = Covariances(estimator="oas").fit_transform(epoch_data.astype(float))
        ts = TangentSpace(metric="riemann").fit_transform(cov)
        feed = ts.astype(np.float32)
    else:
        feed = epoch_data.astype(np.float32)
    p = clf.predict_proba(feed)
    return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()


def resolve_session_posterior(epoch_data: np.ndarray, task: str) -> tuple[np.ndarray, str]:
    """Load available ONNX classifiers for *task* and combine into p_model.

    Uses the mean posterior when both Riemannian and EEGNet are on HF Hub;
    otherwise uses whichever is available. Raises if none load.
    """
    epoch_data = np.asarray(epoch_data)
    specs = _task_models(task)
    posteriors: list[np.ndarray] = []
    tags: list[str] = []

    for name, filename in specs.items():
        try:
            clf = load_onnx_model(filename)
            posteriors.append(_predict_target_proba(clf, epoch_data))
            sha = clf.metadata.get("sha") or clf.metadata.get("model", name)
            tags.append(f"{name}:{sha}")
            log.info("classifier_loaded", name=name, file=filename)
        except Exception as e:
            log.warning("classifier_unavailable", name=name, file=filename, err=str(e))

    if not posteriors:
        raise FileNotFoundError(f"No ONNX classifiers available for task {task!r}")

    combined = np.mean(np.stack(posteriors, axis=0), axis=0)
    model_id = "ensemble(" + "+".join(tags) + ")" if len(posteriors) > 1 else tags[0]
    return combined, model_id
