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
    edf_path: Path = typer.Argument(..., help="Path to an EDF/BDF file"),
    task: str = typer.Option("right_hand", help="Target task"),
    out: Path = typer.Option(Path("mes_result.json"), help="Where to write JSON result"),
) -> None:
    """Score a single recording file and write MES JSON."""
    from mes_core.io import load_eeg
    from mes_core.preprocessing import preprocess_raw, epoch_raw, PreprocessConfig
    from mes_core.features.bandpower import band_power, erd_percent
    from mes_core.scoring import compute_mes, MesWeights, SubjectBaseline
    import numpy as np

    console.print(f"[bold]Loading[/]: {edf_path}")
    raw = load_eeg(str(edf_path))
    raw = preprocess_raw(raw, PreprocessConfig())
    epochs = epoch_raw(raw)
    data = epochs.get_data() if epochs is not None else None
    if data is None or len(data) == 0:
        console.print("[red]No usable epochs found.[/]")
        raise typer.Exit(code=1)

    baseline = SubjectBaseline.zeros(n_features=4)
    weights = MesWeights.default()
    p_model = np.full(len(data), 0.5, dtype=float)
    results = compute_mes(epochs_data=data, sfreq=raw.info["sfreq"],
                          ch_names=raw.info["ch_names"], task=task,
                          baseline=baseline, weights=weights, p_model=p_model)

    out.write_text(json.dumps(results.to_dict(), indent=2))
    console.print(f"[green]Wrote[/] {out}  (mean MES={results.mes_per_trial.mean():.1f})")


if __name__ == "__main__":
    app()
