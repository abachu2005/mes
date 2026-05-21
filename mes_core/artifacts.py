"""Load published MES artifacts (weights, baselines) from package data or HF Hub."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import structlog

from mes_core.scoring import MesWeights, SubjectBaseline

log = structlog.get_logger(__name__)

WEIGHTS_FILES: dict[str, str] = {
    "right_hand": "mes_weights_right_hand.json",
}


@lru_cache(maxsize=8)
def load_mes_weights(task: str = "right_hand") -> MesWeights:
    """Return fitted MES weights for *task*, falling back to defaults."""
    key = "right_hand" if "right" in task else task
    filename = WEIGHTS_FILES.get(key, WEIGHTS_FILES["right_hand"])
    try:
        raw = resources.files("mes_core.data").joinpath(filename).read_text(encoding="utf-8")
        data = json.loads(raw)
        return MesWeights.from_dict(data["weights"])
    except Exception as e:
        log.warning("mes_weights_bundle_missing", task=task, err=str(e))
        return MesWeights.default()


def load_population_baseline(task: str = "right_hand") -> SubjectBaseline:
    """Population rest baseline shipped with weights (optional)."""
    key = "right_hand" if "right" in task else task
    filename = WEIGHTS_FILES.get(key, WEIGHTS_FILES["right_hand"])
    try:
        raw = resources.files("mes_core.data").joinpath(filename).read_text(encoding="utf-8")
        data = json.loads(raw)
        if "population_baseline" in data:
            return SubjectBaseline.from_dict(data["population_baseline"])
    except Exception:
        pass
    return SubjectBaseline.zeros(4)


def weights_bundle_path(task: str = "right_hand") -> Path:
    """Filesystem path to the bundled weights JSON (for editors/scripts)."""
    key = "right_hand" if "right" in task else task
    filename = WEIGHTS_FILES.get(key, WEIGHTS_FILES["right_hand"])
    return Path(__file__).resolve().parent / "data" / filename


def write_weights_bundle(
    path: Path,
    weights: MesWeights,
    *,
    population_baseline: SubjectBaseline | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """Write weights JSON in the canonical bundle format."""
    payload: dict[str, Any] = {
        "task": "right_hand_vs_rest",
        "weights": weights.to_dict(),
        "meta": meta or {},
    }
    if population_baseline is not None:
        payload["population_baseline"] = population_baseline.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
