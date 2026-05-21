"""Tools to export trained classifiers to ONNX.

Two pathways:
- Riemannian + LR: serialize via skl2onnx by training a sklearn pipeline whose
  input is the tangent-space feature vector.
- EEGNet (torch): serialize directly via torch.onnx.export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def export_sklearn_pipeline(
    pipeline: Any,
    n_features: int,
    out_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Export a fitted sklearn pipeline to an ONNX file at `out_path`."""
    from skl2onnx import to_onnx

    initial_type = [("X", _float_tensor_type(n_features))]
    onx = to_onnx(
        pipeline,
        initial_types=initial_type,
        target_opset={"": 17, "ai.onnx.ml": 3},
        options={id(pipeline): {"zipmap": False}},
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onx.SerializeToString())
    if metadata:
        out.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return out


def _float_tensor_type(n_features: int):
    from skl2onnx.common.data_types import FloatTensorType

    return FloatTensorType([None, n_features])


def export_torch_eegnet(
    model: Any,
    n_channels: int,
    n_samples: int,
    out_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Export a Braindecode EEGNet model to ONNX."""
    import torch

    model.eval()
    dummy = torch.zeros(1, n_channels, n_samples, dtype=torch.float32)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "input_names": ["X"],
        "output_names": ["proba"],
        "dynamic_axes": {"X": {0: "batch"}, "proba": {0: "batch"}},
        "opset_version": 17,
    }
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(out), **export_kwargs)
    if metadata:
        out.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return out


def smoke_test_onnx(model_path: str | Path, x: np.ndarray) -> np.ndarray:
    """Load an ONNX model and run one forward pass with `x` to sanity-check."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: x.astype(np.float32)})[0]
    return np.asarray(out)
