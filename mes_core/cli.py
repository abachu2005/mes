"""Command-line interface for MES."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mes_core import __version__
from mes_core.config import HF_REPOS, OPENBCI_MONTAGE_16, TARGET_SFREQ

app = typer.Typer(help="MES — Motor Engagement Signal CLI")
console = Console()


@app.command()
def version() -> None:
    """Print version and key config."""
    table = Table(title=f"MES v{__version__}", show_header=False)
    table.add_row("Target sample rate (Hz)", str(TARGET_SFREQ))
    table.add_row("Channels", ", ".join(OPENBCI_MONTAGE_16))
    table.add_row("HF dataset repo", HF_REPOS.dataset)
    table.add_row("HF model repo", HF_REPOS.model)
    table.add_row("HF Space", HF_REPOS.space)
    console.print(table)


@app.command()
def score(
    edf_path: Path = typer.Argument(..., help="Path to an EDF/BDF/OpenBCI file"),
    task: str = typer.Option("right_hand", help="Target task"),
    out: Path = typer.Option(Path("mes_result.json"), help="Where to write JSON result"),
    no_onnx: bool = typer.Option(False, help="Skip ONNX ensemble (heuristic posterior)"),
) -> None:
    """Score a single recording with production ONNX + fitted MES weights."""
    from mes_core.pipeline import score_recording

    console.print(f"[bold]Loading[/]: {edf_path}")
    result = score_recording(edf_path, task=task, use_onnx=not no_onnx)
    payload = result.to_dict()
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    mean = result.mes.summary["mes_mean"]
    console.print(
        f"[green]Wrote[/] {out}  mean MES={mean:.1f}  model={result.model_sha}  "
        f"baseline={result.baseline_kind}"
    )


@app.command("validate")
def validate_cmd(
    data_dir: Path | None = typer.Option(None, help="Directory of processed parquet files"),
    download: bool = typer.Option(False, "--download", help="Download parquet subset from HF Hub"),
    out_dir: Path = typer.Option(Path("validation_out"), help="Output directory for reports"),
    max_files: int = typer.Option(30, help="Max parquet files to load"),
) -> None:
    """Run MES vs baseline metrics on labeled processed data."""
    from mes_core.eval.parquet import download_processed_cache
    from mes_core.eval.validate import run_validation, write_validation_report

    if download or data_dir is None:
        console.print("[bold]Downloading[/] processed parquet subset…")
        data_dir = download_processed_cache(max_files=max_files)
    assert data_dir is not None
    console.print(f"[bold]Validating[/] {data_dir}")
    report = run_validation(data_dir, max_files=max_files)
    write_validation_report(report, out_dir)
    console.print(f"[green]Wrote[/] {out_dir}/validation_report.json")
    for m in report.models:
        console.print(f"  {m.name}: acc={m.accuracy:.3f} auc={m.auc:.3f}")


@app.command("fit-weights")
def fit_weights_cmd(
    data_dir: Path | None = typer.Option(None, help="Processed parquet directory"),
    download: bool = typer.Option(True, help="Download from HF if no data-dir"),
    max_files: int = typer.Option(40),
    upload: bool = typer.Option(False, help="Upload bundle to HF model repo"),
) -> None:
    """Fit MES weights from parquet and update mes_core/data bundle."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "fit_mes_weights.py"),
        "--max-files",
        str(max_files),
    ]
    if download and data_dir is None:
        cmd.append("--download")
    elif data_dir:
        cmd.extend(["--data-dir", str(data_dir)])
    if upload:
        cmd.append("--upload")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    app()
