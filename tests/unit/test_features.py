"""Feature extraction unit tests."""

from __future__ import annotations

import numpy as np

from mes_core.config import BANDS
from mes_core.features.bandpower import band_power, erd_percent
from mes_core.features.lateralization import default_contra_ipsi_for_task, lateralization_index
from mes_core.features.mrcp import mrcp_features
from mes_core.features.riemann import covariance_features, tangent_space_features


def test_band_power_concentrates_in_target_band() -> None:
    sfreq = 250.0
    t = np.arange(int(2 * sfreq)) / sfreq
    pure_10hz = np.sin(2 * np.pi * 10.0 * t)[None, :]  # (1 channel, n_times)

    p_mu = band_power(pure_10hz, sfreq, BANDS["mu"])
    p_beta = band_power(pure_10hz, sfreq, BANDS["beta"])
    assert p_mu[0] > 10 * p_beta[0]


def test_erd_percent_positive_when_power_drops() -> None:
    sfreq = 250.0
    t = np.arange(int(2 * sfreq)) / sfreq
    base = np.sin(2 * np.pi * 10.0 * t)[None, :]
    task = 0.3 * np.sin(2 * np.pi * 10.0 * t)[None, :]
    erd = erd_percent(task, base, sfreq, BANDS["mu"])
    assert erd[0] > 50.0  # ~91% power drop


def test_erd_percent_negative_when_power_rises() -> None:
    sfreq = 250.0
    t = np.arange(int(2 * sfreq)) / sfreq
    base = np.sin(2 * np.pi * 10.0 * t)[None, :]
    task = 2.0 * np.sin(2 * np.pi * 10.0 * t)[None, :]
    erd = erd_percent(task, base, sfreq, BANDS["mu"])
    assert erd[0] < -50.0


def test_lateralization_index_sign() -> None:
    ch_names = ["C3", "C4", "FC3", "FC4"]
    erd = np.array([60.0, 10.0, 50.0, 20.0])  # strong contra, weak ipsi
    contra, ipsi = default_contra_ipsi_for_task("right_hand")
    li = lateralization_index(erd, ch_names, contra_channels=contra, ipsi_channels=ipsi)
    assert li > 0  # contra > ipsi


def test_lateralization_handles_missing_channels() -> None:
    ch_names = ["C3"]
    erd = np.array([60.0])
    contra, ipsi = default_contra_ipsi_for_task("right_hand")
    li = lateralization_index(erd, ch_names, contra_channels=contra, ipsi_channels=ipsi)
    assert np.isnan(li)


def test_mrcp_amplitude_signs_with_negative_deflection() -> None:
    sfreq = 125.0
    tmin = -2.0
    n = round((4.0 - tmin) * sfreq)
    times = tmin + np.arange(n) / sfreq
    sig = np.where(times > 0, -1.5 * np.exp(-times.clip(0)), 0.0)
    epochs = sig[None, None, :]  # (1 epoch, 1 ch, n_times)
    feat = mrcp_features(epochs, sfreq, tmin=tmin)
    assert feat["amplitude"][0, 0] < 0


def test_riemann_covariance_and_tangent_shapes() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 8, 250))
    C = covariance_features(X)
    assert C.shape == (20, 8, 8)
    ts = tangent_space_features(C)
    assert ts.shape == (20, 8 * (8 + 1) // 2)


def test_csp_runs_on_synthetic_binary() -> None:
    from mes_core.features.csp import apply_csp, fit_csp

    rng = np.random.default_rng(1)
    n = 30
    X0 = rng.standard_normal((n, 8, 256))
    X1 = rng.standard_normal((n, 8, 256)) * 2.0
    X = np.concatenate([X0, X1])
    y = np.array([0] * n + [1] * n)

    csp = fit_csp(X, y, n_components=4)
    feats = apply_csp(csp, X)
    assert feats.shape == (2 * n, 4)
