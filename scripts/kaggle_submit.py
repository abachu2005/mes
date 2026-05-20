"""Headless Kaggle Notebook submitter.

Renders a `.py` source via jupytext into a `.ipynb`, writes a
`kernel-metadata.json` next to it, calls `kaggle kernels push`, polls
status, then pulls outputs.

Usage:
    python scripts/kaggle_submit.py notebooks/kaggle/00_preprocess.py \\
        --kernel-id "abachu2005/mes-preprocess" --gpu false --internet true
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

KAGGLE_CLI = os.environ.get("KAGGLE_CLI") or shutil.which("kaggle") or "kaggle"

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _env_for_kaggle() -> dict[str, str]:
    """Build env with Kaggle credentials from .env.local or process env."""
    from dotenv import load_dotenv  # type: ignore[import-untyped]

    env_path = Path(".env.local")
    if env_path.exists():
        load_dotenv(env_path)

    env = os.environ.copy()
    new_style = env.get("KAGGLE_API_TOKEN")
    user = env.get("KAGGLE_USERNAME")
    key = env.get("KAGGLE_KEY")

    if new_style and not (user and key):
        # The kaggle CLI traditionally needs username+key in ~/.kaggle/kaggle.json.
        # New-style tokens can also be stored in ~/.kaggle/access_token.
        home_kaggle = Path.home() / ".kaggle"
        home_kaggle.mkdir(exist_ok=True)
        access_path = home_kaggle / "access_token"
        access_path.write_text(new_style)
        try:
            os.chmod(access_path, 0o600)
        except Exception:
            pass
        env["KAGGLE_API_TOKEN"] = new_style
    elif user and key:
        home_kaggle = Path.home() / ".kaggle"
        home_kaggle.mkdir(exist_ok=True)
        (home_kaggle / "kaggle.json").write_text(
            json.dumps({"username": user, "key": key})
        )
        try:
            os.chmod(home_kaggle / "kaggle.json", 0o600)
        except Exception:
            pass

    return env


def _render_notebook(py_path: Path) -> Path:
    """Convert a jupytext .py script to .ipynb in a sibling build/ dir."""
    import jupytext  # type: ignore[import-untyped]

    build = py_path.parent / "build"
    build.mkdir(exist_ok=True)
    nb_path = build / (py_path.stem + ".ipynb")
    nb = jupytext.read(str(py_path))
    nb.metadata.setdefault("kernelspec", {})
    nb.metadata["kernelspec"].update({
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    })
    nb.metadata.setdefault("language_info", {})
    nb.metadata["language_info"].update({"name": "python"})
    jupytext.write(nb, str(nb_path), fmt="ipynb")
    return nb_path


def _write_metadata(
    nb_path: Path,
    kernel_id: str,
    *,
    enable_gpu: bool,
    enable_internet: bool,
    secrets: list[str],
    kernel_sources: list[str],
) -> Path:
    meta = {
        "id": kernel_id,
        "title": Path(kernel_id).name.replace("-", " ").title(),
        "code_file": nb_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": enable_internet,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": kernel_sources,
        "model_sources": [],
    }
    # The kaggle CLI looks for kernel-metadata.json in the same folder as code_file.
    meta_path = nb_path.parent / "kernel-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


@app.command()
def push(
    source: Path = typer.Argument(..., help="Path to a jupytext .py notebook source"),
    kernel_id: str = typer.Option(..., "--kernel-id", help="username/kernel-slug"),
    gpu: bool = typer.Option(False, help="Request a GPU"),
    internet: bool = typer.Option(True, help="Enable internet"),
    secret: list[str] = typer.Option([], help="Secret names already configured on Kaggle"),
    kernel_source: list[str] = typer.Option([], "--kernel-source", help="Upstream kernel slug(s)"),
    poll: bool = typer.Option(True, help="Poll until completion"),
    poll_interval_s: int = typer.Option(60, help="Poll interval"),
    poll_timeout_min: int = typer.Option(360, help="Max minutes to poll"),
    pull: bool = typer.Option(True, help="Download outputs after completion"),
) -> None:
    """Push a notebook to Kaggle and (optionally) wait for it to finish."""
    env = _env_for_kaggle()
    nb_path = _render_notebook(source)
    _write_metadata(
        nb_path, kernel_id,
        enable_gpu=gpu, enable_internet=internet, secrets=secret,
        kernel_sources=kernel_source,
    )

    console.print(f"[bold]Pushing[/] {kernel_id} from {nb_path}")
    res = subprocess.run(
        [KAGGLE_CLI, "kernels", "push", "-p", str(nb_path.parent)],
        env=env, text=True, capture_output=True,
    )
    console.print(res.stdout)
    if res.returncode != 0:
        console.print(f"[red]push failed[/]: {res.stderr}")
        raise typer.Exit(code=1)

    if not poll:
        return

    deadline = time.time() + poll_timeout_min * 60
    while time.time() < deadline:
        s = subprocess.run(
            [KAGGLE_CLI, "kernels", "status", kernel_id],
            env=env, text=True, capture_output=True,
        )
        line = s.stdout.strip().splitlines()[-1] if s.stdout.strip() else ""
        console.print(f"[dim]{time.strftime('%H:%M:%S')}[/] {line}")
        ll = line.lower()
        if "complete" in ll or "succeeded" in ll:
            break
        if "error" in ll or "failed" in ll or "cancel" in ll:
            console.print(f"[red]kernel failed[/]: {line}")
            sys.exit(2)
        time.sleep(poll_interval_s)

    if pull:
        out_dir = nb_path.parent / "output"
        out_dir.mkdir(exist_ok=True)
        subprocess.run(
            [KAGGLE_CLI, "kernels", "output", kernel_id, "-p", str(out_dir)],
            env=env, text=True,
        )
        console.print(f"[green]pulled outputs to[/] {out_dir}")


if __name__ == "__main__":
    app()
