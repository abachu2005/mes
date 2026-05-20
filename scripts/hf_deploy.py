"""Push the repo to a Hugging Face Space (Docker SDK)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO = os.environ.get("HF_SPACE_REPO", "abachu2005/mes")
TOKEN = os.environ.get("HF_TOKEN")

SPACE_README = """\
---
title: MES — Motor Engagement Signal
emoji: 🧠
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: apache-2.0
---

# Motor Engagement Signal

Quantifying neural drive for movement recovery from EEG.

See the [GitHub repo](https://github.com/) for full source, methods, and benchmarks.
"""

IGNORE = {
    ".git", ".venv", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "htmlcov", "tests/e2e/node_modules",
    "tests/e2e/test-results", "tests/e2e/playwright-report",
    "notebooks/kaggle/build", "notebooks/kaggle/output",
    "data", "models", "mlruns", ".env", ".env.local",
    "backend/app/static",  # rebuilt by Dockerfile
}


def _walk(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        parts = set(rel.parts)
        if parts & IGNORE or any(seg.startswith(".") and seg != ".dockerignore" and seg != ".gitignore" for seg in rel.parts):
            continue
        if rel.suffix in {".onnx", ".edf", ".bdf", ".fif"}:
            continue
        yield rel


def main() -> int:
    if not TOKEN:
        print("HF_TOKEN missing", file=sys.stderr)
        return 1
    api = HfApi(token=TOKEN)
    create_repo(repo_id=REPO, repo_type="space", space_sdk="docker", exist_ok=True, token=TOKEN)

    # Write/overwrite Space-style README.md inside a temp staging copy.
    root = Path(__file__).resolve().parents[1]
    (root / "_space_README.md").write_text(SPACE_README)

    print(f"Uploading repo to space {REPO}...")
    api.upload_folder(
        folder_path=str(root),
        repo_id=REPO,
        repo_type="space",
        commit_message="deploy MES Space",
        ignore_patterns=[
            ".git/**", ".venv/**", "**/node_modules/**", "**/__pycache__/**",
            "**/.pytest_cache/**", "**/.ruff_cache/**", "**/.mypy_cache/**",
            "htmlcov/**", ".coverage", ".env", ".env.local",
            "tests/e2e/test-results/**", "tests/e2e/playwright-report/**",
            "notebooks/kaggle/build/**", "notebooks/kaggle/output/**",
            "backend/app/static/**", "data/**", "models/**", "mlruns/**",
            "*.onnx", "*.edf", "*.bdf", "*.fif",
        ],
    )
    # README must be at the root of the space and contain front-matter.
    api.upload_file(
        path_or_fileobj=str(root / "_space_README.md"),
        path_in_repo="README.md",
        repo_id=REPO,
        repo_type="space",
        commit_message="set Space README front matter",
    )
    (root / "_space_README.md").unlink(missing_ok=True)
    print(f"https://huggingface.co/spaces/{REPO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
