"""Load processed parquet trials from HF Hub or a local directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_T = 750
LABEL_MAP = {"right_hand": 1, "rest": 0, "left_hand": 1}


@dataclass
class ParquetTrial:
    X: np.ndarray  # (n_channels, n_times)
    y: int
    label: str
    subject: str
    source_file: str


def _reshape_epoch(row: pd.Series) -> np.ndarray:
    x = np.frombuffer(row["data"], dtype="float32").reshape(row["n_channels"], row["n_times"])
    n = x.shape[1]
    if n >= TARGET_T:
        return x[:, :TARGET_T]
    pad = np.zeros((x.shape[0], TARGET_T - n), dtype=np.float32)
    return np.concatenate([x, pad], axis=1)


def load_parquet_dir(
    data_dir: Path,
    *,
    prefix: str = "physionet_",
    labels: set[str] | None = None,
    max_files: int | None = None,
) -> list[ParquetTrial]:
    """Load trials from ``*.parquet`` in *data_dir*."""
    if labels is None:
        labels = set(LABEL_MAP)
    files = sorted(data_dir.glob("*.parquet"))
    if prefix:
        files = [f for f in files if f.name.startswith(prefix)]
    if max_files:
        files = files[:max_files]
    rows: list[ParquetTrial] = []
    for f in files:
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            lab = str(r["label"])
            if lab not in labels:
                continue
            y = LABEL_MAP.get(lab)
            if y is None:
                continue
            rows.append(
                ParquetTrial(
                    X=_reshape_epoch(r),
                    y=int(y),
                    label=lab,
                    subject=str(r["subject"]),
                    source_file=f.name,
                )
            )
    return rows


def download_processed_cache(
    *,
    max_files: int = 40,
    cache_dir: Path | None = None,
) -> Path:
    """Download a subset of processed parquet files into a local cache directory."""
    from huggingface_hub import HfApi, hf_hub_download

    from mes_core.config import HF_REPOS, cache_root

    out = cache_dir or (cache_root() / "eval_parquet")
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = [
        f
        for f in api.list_repo_files(HF_REPOS.dataset, repo_type="dataset")
        if f.startswith("processed/") and f.endswith(".parquet")
    ]
    files = sorted(files)[:max_files]
    for remote in files:
        name = Path(remote).name
        dest = out / name
        if dest.exists():
            continue
        local = hf_hub_download(
            repo_id=HF_REPOS.dataset,
            filename=remote,
            repo_type="dataset",
        )
        dest.write_bytes(Path(local).read_bytes())
    return out
