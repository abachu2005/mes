"""Link MES scores to external clinical outcome tables (CSV)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mes_core.eval.metrics import spearman_corr


def load_outcomes_csv(path: Path) -> dict[str, dict[str, float]]:
    """CSV columns: participant_code, session_index (optional), fma, arat, ..."""
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("participant_code") or row.get("code")
            if not code:
                continue
            metrics = {}
            for k, v in row.items():
                if k in ("participant_code", "code", "session_index", "session_id"):
                    continue
                try:
                    metrics[k] = float(v)
                except (TypeError, ValueError):
                    continue
            rows[code] = metrics
    return rows


def correlate_mes_with_outcomes(
    mes_by_participant: dict[str, list[float]],
    outcomes: dict[str, dict[str, float]],
    *,
    outcome_key: str = "fma",
) -> dict[str, float]:
    """Spearman correlation between mean MES and clinical metric per participant."""
    mes_vals, out_vals = [], []
    for code, mes_list in mes_by_participant.items():
        if code not in outcomes or outcome_key not in outcomes[code]:
            continue
        if not mes_list:
            continue
        mes_vals.append(float(np.mean(mes_list)))
        out_vals.append(outcomes[code][outcome_key])
    if len(mes_vals) < 3:
        return {"n": len(mes_vals), "spearman_rho": float("nan")}
    return {
        "n": len(mes_vals),
        "spearman_rho": spearman_corr(np.array(mes_vals), np.array(out_vals)),
        "outcome_key": outcome_key,
    }
