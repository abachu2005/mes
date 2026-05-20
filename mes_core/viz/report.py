"""Session-report HTML + PDF rendering.

Renders a self-contained HTML report from a Jinja2 template and converts
it to PDF via WeasyPrint. Matplotlib supplies the static figures (MES gauge,
time trace, ERD topomap) since WeasyPrint doesn't run JavaScript.
"""

from __future__ import annotations

import base64
import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "backend" / "app" / "templates"


@dataclass
class ReportContext:
    """Data passed to the report template."""

    participant_code: str
    session_id: str
    task: str
    target_limb: str
    headset: str
    mes_mean: float
    mes_median: float
    mes_std: float
    n_trials: int
    mes_per_trial: list[float]
    erd_topomap: dict[str, Any]
    lateralization: float
    model_sha: str
    created_at: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fig_to_data_uri(fig) -> str:  # type: ignore[no-untyped-def]
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _gauge_figure(value: float):  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    value = float(np.clip(value, 0.0, 100.0))
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    theta = np.linspace(0, np.pi, 200)
    ax.barh(1.0, np.pi, left=0.0, height=0.25, color="#e6eaf2", edgecolor="none")
    ax.barh(1.0, value / 100.0 * np.pi, left=0.0, height=0.25, color="#0d9488", edgecolor="none")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_frame_on(False)
    ax.text(np.pi / 2, 0.0, f"{value:.1f}", ha="center", va="center",
            fontsize=22, fontweight="bold", color="#0f172a")
    ax.text(np.pi / 2, -0.5, "MES", ha="center", va="center", fontsize=10, color="#475569")
    fig.tight_layout()
    return fig


def _trace_figure(values: np.ndarray):  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(np.arange(len(values)) + 1, values, marker="o",
            color="#0d9488", linewidth=2)
    ax.set_xlabel("Trial")
    ax.set_ylabel("MES")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.axhline(50, color="#94a3b8", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    return fig


def _topomap_figure(payload: dict[str, Any]):  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    pts = payload.get("points", [])
    if not pts:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        return fig
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    vs = [p["value"] for p in pts]
    sc = ax.scatter(xs, ys, c=vs, cmap="RdBu_r", s=300, edgecolors="#0f172a", linewidths=1.5)
    for p in pts:
        ax.text(p["x"], p["y"], p["channel"], ha="center", va="center",
                fontsize=6, color="#0f172a")
    circle = plt.Circle((0, 0), 1.0, fill=False, color="#94a3b8", linewidth=1.5)
    ax.add_patch(circle)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="ERD%")
    fig.tight_layout()
    return fig


def build_session_report_html(ctx: ReportContext) -> str:
    """Render the session report HTML using Jinja2."""
    import matplotlib

    matplotlib.use("Agg")  # ensure headless
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    gauge_img = _fig_to_data_uri(_gauge_figure(ctx.mes_mean))
    trace_img = _fig_to_data_uri(_trace_figure(np.array(ctx.mes_per_trial)))
    topo_img = _fig_to_data_uri(_topomap_figure(ctx.erd_topomap))

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    tpl = env.get_template("session_report.html")
    return tpl.render(
        ctx=ctx,
        gauge_img=gauge_img,
        trace_img=trace_img,
        topo_img=topo_img,
    )


def render_pdf(html: str, out_path: str | Path) -> Path:
    """Render an HTML string to PDF using WeasyPrint."""
    from weasyprint import HTML  # type: ignore[import-untyped]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf(str(out))
    return out
