"""Common Spatial Patterns (CSP) features for motor imagery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class CspFilters:
    filters: np.ndarray  # (n_components, n_channels)
    patterns: np.ndarray


def fit_csp(X: np.ndarray, y: np.ndarray, *, n_components: int = 4) -> CspFilters:
    """Fit CSP spatial filters for binary classes.

    X shape: (n_epochs, n_channels, n_times)
    """
    from scipy.linalg import eigh

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("CSP requires exactly two classes")

    covs = []
    for c in classes:
        trials = X[y == c]
        cov = np.mean([np.cov(trial) for trial in trials], axis=0)
        covs.append(cov)
    cov_a, cov_b = covs
    reg = 1e-6 * np.eye(cov_a.shape[0])
    cov_a, cov_b = cov_a + reg, cov_b + reg
    eigvals, eigvecs = eigh(cov_a, cov_a + cov_b)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    half = max(1, n_components // 2)
    half = min(half, eigvecs.shape[1] // 2)
    filters = np.concatenate([eigvecs[:, :half], eigvecs[:, -half:]], axis=1).T
    return CspFilters(filters=filters, patterns=eigvecs[:, :half].T)


def apply_csp(csp: CspFilters, X: np.ndarray) -> np.ndarray:
    """Project trials through CSP filters; return log-variance features."""
    X = np.asarray(X, dtype=float)
    feats = []
    for trial in X:
        projected = csp.filters @ trial
        feats.append(np.log(np.var(projected, axis=-1) + 1e-12))
    return np.stack(feats)


def csp_lda_loso_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, list[float]]:
    """LOSO accuracy with CSP log-variance features + LDA."""
    csp = fit_csp(X, y, n_components=4)
    feats = apply_csp(csp, X)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    accs: list[float] = []
    for tr, te in gkf.split(feats, y, groups=groups):
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis()),
        ])
        pipe.fit(feats[tr], y[tr])
        accs.append(float(pipe.score(feats[te], y[te])))
    return float(np.mean(accs)), accs
