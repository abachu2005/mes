"""MES scoring tests."""

from __future__ import annotations

import numpy as np

from mes_core.config import OPENBCI_MONTAGE_16
from mes_core.eval.metrics import brier_score, cohen_d, icc_3_1, spearman_corr
from mes_core.scoring import (
    MesWeights,
    SubjectBaseline,
    compute_mes,
    fit_mes_weights,
    fit_subject_baseline,
)


def _make_engagement_epochs(n_trials, sfreq=125.0, duration=4.0, mu_drop=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n_ch = len(OPENBCI_MONTAGE_16)
    n_times = int(sfreq * duration)
    t = np.arange(n_times) / sfreq
    out = np.empty((n_trials, n_ch, n_times))
    contra_idx = OPENBCI_MONTAGE_16.index("C3")  # right-hand task -> contra C3

    for trial in range(n_trials):
        for ch in range(n_ch):
            sig = (
                10.0 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
                + 4.0 * np.sin(2 * np.pi * 20.0 * t + rng.uniform(0, 2 * np.pi))
                + rng.standard_normal(n_times) * 0.5
            )
            # Simulate post-stim ERD on contra
            if ch == contra_idx:
                second_half = slice(n_times // 2, None)
                sig[second_half] *= mu_drop
            out[trial, ch] = sig
    return out


def test_mes_is_bounded_0_100() -> None:
    epochs = _make_engagement_epochs(n_trials=8)
    bl = SubjectBaseline.zeros(4)
    w = MesWeights.default()
    p = np.full(8, 0.5)
    res = compute_mes(
        epochs_data=epochs, sfreq=125.0, ch_names=list(OPENBCI_MONTAGE_16),
        task="right_hand", baseline=bl, weights=w, p_model=p,
    )
    assert res.mes_per_trial.shape == (8,)
    assert np.all((res.mes_per_trial >= 0) & (res.mes_per_trial <= 100))


def test_mes_is_higher_for_engaged_than_rest() -> None:
    engaged = _make_engagement_epochs(n_trials=15, mu_drop=0.2, seed=1)
    rest = _make_engagement_epochs(n_trials=15, mu_drop=1.0, seed=2)

    bl = SubjectBaseline.zeros(4)
    w = MesWeights.default()
    p_eng = np.full(len(engaged), 0.9)
    p_rest = np.full(len(rest), 0.1)

    res_e = compute_mes(engaged, 125.0, list(OPENBCI_MONTAGE_16), "right_hand", bl, w, p_eng)
    res_r = compute_mes(rest, 125.0, list(OPENBCI_MONTAGE_16), "right_hand", bl, w, p_rest)
    assert res_e.summary["mes_mean"] > res_r.summary["mes_mean"]


def test_subject_baseline_serialization_roundtrip() -> None:
    bl = SubjectBaseline(
        feature_names=("z_mu", "z_beta", "z_li", "z_mrcp"),
        mean=np.array([1.0, 2.0, 3.0, 4.0]),
        std=np.array([0.1, 0.2, 0.3, 0.4]),
    )
    d = bl.to_dict()
    bl2 = SubjectBaseline.from_dict(d)
    assert bl.feature_names == bl2.feature_names
    assert np.allclose(bl.mean, bl2.mean)
    assert np.allclose(bl.std, bl2.std)


def test_fit_mes_weights_recovers_signal() -> None:
    rng = np.random.default_rng(0)
    n = 200
    z = rng.standard_normal((n, 4))
    p = rng.uniform(0.05, 0.95, n)
    # Simulate labels driven by sum of first feature + logit(p)
    logits = 1.5 * z[:, 0] + 1.0 * (np.log(p / (1 - p)))
    labels = (logits + rng.normal(0, 0.5, n) > 0).astype(int)

    w = fit_mes_weights(z, p, labels)
    assert w.w_mu > 0.5  # picked up the signal
    assert w.w_pmodel > 0.5


def test_fit_subject_baseline_returns_correct_shape() -> None:
    rest = _make_engagement_epochs(n_trials=5, mu_drop=1.0)
    bl = fit_subject_baseline(rest, 125.0, list(OPENBCI_MONTAGE_16), "right_hand")
    assert bl.mean.shape == (4,)
    assert bl.std.shape == (4,)
    assert np.all(bl.std > 0)


def test_icc_perfect_when_no_session_variance() -> None:
    # 5 subjects, 3 sessions, each subject has stable value
    subj_vals = np.array([10, 20, 30, 40, 50.0])
    vals = np.tile(subj_vals[:, None], (1, 3))
    icc = icc_3_1(vals)
    assert icc > 0.99


def test_brier_score_perfect_is_zero() -> None:
    bs = brier_score(np.array([0.0, 1.0, 0.0, 1.0]), np.array([0, 1, 0, 1]))
    assert abs(bs) < 1e-9


def test_cohen_d_simple() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(2.0, 1.0, 50)
    b = rng.normal(0.0, 1.0, 50)
    assert cohen_d(a, b) > 0.5


def test_spearman_corr_monotonic() -> None:
    a = np.arange(20.0)
    b = a ** 2  # monotonic
    rho = spearman_corr(a, b)
    assert rho > 0.99
