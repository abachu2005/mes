"""Validation metrics used across the eval suite and benchmarks.md.

Implemented from numpy/scipy directly so the code does not depend on heavier
statsmodels installs on the inference side.
"""

from __future__ import annotations

import numpy as np


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size for two paired samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    if diff.std(ddof=1) == 0:
        return 0.0
    return float(diff.mean() / diff.std(ddof=1))


def icc_3_1(values: np.ndarray) -> float:
    """Intraclass correlation, ICC(3,1) - two-way mixed, single rater.

    `values` is (n_subjects, n_sessions).
    """
    values = np.asarray(values, dtype=float)
    n, k = values.shape
    if n < 2 or k < 2:
        return float("nan")
    mean_per_subject = values.mean(axis=1, keepdims=True)
    mean_per_session = values.mean(axis=0, keepdims=True)
    grand_mean = values.mean()

    msr = ((mean_per_subject - grand_mean) ** 2).sum() * k / (n - 1)
    msc = ((mean_per_session - grand_mean) ** 2).sum() * n / (k - 1)
    mse_num = ((values - mean_per_subject - mean_per_session + grand_mean) ** 2).sum()
    mse = mse_num / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom <= 0:
        return float("nan")
    return float((msr - mse) / denom)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided paired Wilcoxon signed-rank, returns (statistic, p)."""
    from scipy.stats import wilcoxon

    res = wilcoxon(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return float(res.statistic), float(res.pvalue)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    rho = spearmanr(a, b).correlation
    return float(rho) if rho is not None else float("nan")
