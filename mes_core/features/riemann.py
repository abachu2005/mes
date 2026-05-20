"""Riemannian covariance + tangent-space features."""

from __future__ import annotations

import numpy as np


def covariance_features(epochs_data: np.ndarray, *, estimator: str = "oas") -> np.ndarray:
    """Compute per-epoch spatial covariance matrices.

    Parameters
    ----------
    epochs_data : array, shape (n_epochs, n_channels, n_times)
    estimator : 'oas' (Oracle Approximating Shrinkage, robust) or 'scm'.

    Returns
    -------
    cov : array, shape (n_epochs, n_channels, n_channels)
    """
    from pyriemann.estimation import Covariances

    cov = Covariances(estimator=estimator)
    out: np.ndarray = cov.transform(np.asarray(epochs_data, dtype=float))
    return out


def tangent_space_features(cov: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Project covariance matrices to the tangent space at `reference` (or Riemannian mean)."""
    from pyriemann.tangentspace import TangentSpace

    ts = TangentSpace(metric="riemann")
    if reference is None:
        ts.fit(cov)
    else:
        ts.reference_ = reference
    out: np.ndarray = ts.transform(cov)
    return out
