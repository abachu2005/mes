"""Tests for ONNX classifier routing (Riemannian + EEGNet ensemble)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mes_core.models.inference import OnnxClassifier, resolve_session_posterior


def _fake_clf(*, tangent: bool, proba: np.ndarray, tag: str) -> OnnxClassifier:
    return OnnxClassifier(
        session=MagicMock(),
        input_name="X",
        output_name="proba",
        n_classes=2,
        feature_shape=(136,) if tangent else (16, 750),
        metadata={"model": tag},
    )


@patch("mes_core.models.inference.load_onnx_model")
@patch("mes_core.models.inference._predict_target_proba")
def test_ensemble_averages_both_models(mock_predict, mock_load) -> None:
    riem = _fake_clf(tangent=True, proba=np.array([0.2, 0.4]), tag="riemannian_lr")
    eeg = _fake_clf(tangent=False, proba=np.array([0.6, 0.8]), tag="eegnet_v4")
    mock_load.side_effect = [riem, eeg]
    mock_predict.side_effect = [np.array([0.2, 0.4]), np.array([0.6, 0.8])]

    p, model_id = resolve_session_posterior(np.zeros((2, 16, 750)), "right_hand")

    assert np.allclose(p, [0.4, 0.6])
    assert "ensemble" in model_id
    assert "riemannian" in model_id
    assert "eegnet" in model_id


@patch("mes_core.models.inference.load_onnx_model")
@patch("mes_core.models.inference._predict_target_proba")
def test_single_model_when_eegnet_missing(mock_predict, mock_load) -> None:
    riem = _fake_clf(tangent=True, proba=np.array([0.3, 0.7]), tag="riemannian_lr")
    mock_load.side_effect = [riem, FileNotFoundError("missing")]
    mock_predict.return_value = np.array([0.3, 0.7])

    p, model_id = resolve_session_posterior(np.zeros((2, 16, 750)), "right_hand")

    assert np.allclose(p, [0.3, 0.7])
    assert model_id.startswith("riemannian:")


@patch("mes_core.models.inference.load_onnx_model")
def test_raises_when_no_models(mock_load) -> None:
    mock_load.side_effect = FileNotFoundError("missing")
    with pytest.raises(FileNotFoundError):
        resolve_session_posterior(np.zeros((1, 16, 750)), "right_hand")
