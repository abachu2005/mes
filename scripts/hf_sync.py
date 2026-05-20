#!/usr/bin/env python3
"""Upload Kaggle kernel output (or any local folder) to Hugging Face Hub.

Used by GitHub Actions after `kaggle kernels output` — Kaggle never talks to HF.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync a local folder to Hugging Face Hub")
    parser.add_argument("--local-dir", required=True, help="Root dir from `kaggle kernels output`")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument(
        "--subdir",
        default="",
        help="Upload only this subdirectory (e.g. processed). Default: entire local-dir.",
    )
    parser.add_argument("--message", default="sync from kaggle")
    parser.add_argument("--pattern", default="", help="Optional glob, e.g. '*.onnx'")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi, create_repo

    root = Path(args.local_dir)
    upload_path = root / args.subdir if args.subdir else root
    if not upload_path.exists():
        # Kaggle may flatten: look for subdir anywhere under root
        if args.subdir:
            matches = list(root.rglob(args.subdir))
            for m in matches:
                if m.is_dir():
                    upload_path = m
                    break
        if not upload_path.exists():
            print(f"Upload path not found: {upload_path}", file=sys.stderr)
            print("Contents of local-dir:", list(root.rglob("*"))[:30], file=sys.stderr)
            return 1

    api = HfApi(token=token)
    create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True, token=token)

    if args.pattern:
        files = sorted(upload_path.rglob(args.pattern.lstrip("/")))
        if not files:
            print(f"No files matching {args.pattern!r} in {upload_path}", file=sys.stderr)
            return 1
        for f in files:
            rel = f.relative_to(upload_path)
            print(f"  upload {rel} -> {args.repo_id}")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=str(rel),
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                commit_message=args.message,
            )
    else:
        n = len(list(upload_path.rglob("*")))
        print(f"Uploading {upload_path} ({n} entries) -> {args.repo_id} ({args.repo_type})")
        api.upload_folder(
            folder_path=str(upload_path),
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=args.message,
        )

    print(f"OK: https://huggingface.co/{'datasets' if args.repo_type == 'dataset' else 'models'}/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
