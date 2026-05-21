"""Hugging Face Hub I/O for processed datasets and trained models.

Provides:
- `download_processed_split(name)` to grab processed parquet from the dataset repo
- `download_model(name)` to grab an ONNX model from the model repo
- `upload_processed_dataset(local_dir)` used by the Kaggle preprocess notebook
- `upload_models(local_dir)` used by training notebooks
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import structlog

from mes_core.config import HF_REPOS, PINNED_DATASET_SHA, PINNED_MODEL_SHA, cache_root

log = structlog.get_logger(__name__)


def _hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def download_processed_split(
    split_name: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download a processed dataset split (e.g. 'physionet_train.parquet')."""
    from huggingface_hub import hf_hub_download

    rev = revision or PINNED_DATASET_SHA or "main"
    local = hf_hub_download(
        repo_id=HF_REPOS.dataset,
        filename=split_name,
        repo_type="dataset",
        revision=rev,
        cache_dir=str(cache_dir) if cache_dir else str(cache_root() / "datasets"),
        token=_hf_token(),
    )
    return Path(local)


def download_model(
    model_filename: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download a model ONNX file from the model repo."""
    from huggingface_hub import hf_hub_download

    rev = revision or PINNED_MODEL_SHA or "main"
    try:
        local = hf_hub_download(
            repo_id=HF_REPOS.model,
            filename=model_filename,
            revision=rev,
            cache_dir=str(cache_dir) if cache_dir else str(cache_root() / "models"),
            token=_hf_token(),
        )
        return Path(local)
    except Exception as e:
        log.warning("hf_model_download_failed", filename=model_filename, error=str(e))
        raise


def upload_processed_dataset(local_dir: str | Path, *, commit_message: str = "publish processed split") -> str:
    """Upload a directory of processed parquet files to the dataset repo."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=_hf_token())
    create_repo(repo_id=HF_REPOS.dataset, repo_type="dataset", exist_ok=True, token=_hf_token())
    local = Path(local_dir)
    path_in_repo = "processed" if local.name == "processed" else ""
    commit = api.upload_folder(
        folder_path=str(local),
        path_in_repo=path_in_repo,
        repo_id=HF_REPOS.dataset,
        repo_type="dataset",
        commit_message=commit_message,
    )
    return commit.oid if hasattr(commit, "oid") else str(commit)


def upload_models(local_dir: str | Path, *, commit_message: str = "publish models") -> str:
    """Upload trained ONNX model artifacts to the model repo."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=_hf_token())
    create_repo(repo_id=HF_REPOS.model, exist_ok=True, token=_hf_token())
    commit = api.upload_folder(
        folder_path=str(local_dir),
        repo_id=HF_REPOS.model,
        commit_message=commit_message,
    )
    return commit.oid if hasattr(commit, "oid") else str(commit)


def upload_session_files(
    local_paths: Iterable[str | Path],
    *,
    repo_subdir: str = "",
    commit_message: str = "store session artifact",
) -> str:
    """Push uploaded EEG files + generated reports to the sessions repo."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=_hf_token())
    create_repo(
        repo_id=HF_REPOS.sessions,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=_hf_token(),
    )
    last_oid = ""
    for p in local_paths:
        p = Path(p)
        path_in_repo = f"{repo_subdir.rstrip('/')}/{p.name}" if repo_subdir else p.name
        commit = api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=path_in_repo,
            repo_id=HF_REPOS.sessions,
            repo_type="dataset",
            commit_message=commit_message,
        )
        last_oid = commit.oid if hasattr(commit, "oid") else str(commit)
    return last_oid
