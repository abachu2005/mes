"""Common Spatial Patterns (CSP) for binary motor-imagery classification."""

from __future__ import annotations

from typing import Any

import numpy as np


def fit_csp(X: np.ndarray, y: np.ndarray, *, n_components: int = 6) -> Any:
    """Fit MNE's CSP. Returns the fitted estimator."""
    from mne.decoding import CSP

    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    csp.fit(np.asarray(X, dtype=float), np.asarray(y).astype(int))
    return csp


def apply_csp(csp: Any, X: np.ndarray) -> np.ndarray:
    """Apply a fitted CSP transform to new epochs."""
    out: np.ndarray = csp.transform(np.asarray(X, dtype=float))
    return out
