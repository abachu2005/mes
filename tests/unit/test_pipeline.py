"""Session scoring pipeline."""

from __future__ import annotations

import numpy as np

from mes_core.config import OPENBCI_MONTAGE_16
from mes_core.pipeline import score_epochs
from tests.unit.test_mes import _make_engagement_epochs


def test_score_epochs_bounded_with_fitted_weights() -> None:
    engaged = _make_engagement_epochs(n_trials=6, mu_drop=0.25)
    result = score_epochs(
        engaged,
        sfreq=125.0,
        ch_names=list(OPENBCI_MONTAGE_16),
        task="right_hand",
        use_onnx=False,
    )
    assert np.all((result.mes.mes_per_trial >= 0) & (result.mes.mes_per_trial <= 100))
    assert result.model_sha == "no_model"
