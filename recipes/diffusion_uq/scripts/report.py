#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Comparison report for the diffusion_uq recipe.

Three sub-reports:

1. **Baseline comparison** — final test-set L2 errors of the diffusion ensemble
   (``loss/chunked_test/<field>_uq_mean_l2err/last``, 4 Euler steps × 6 draws)
   versus the deterministic regression baseline
   (``loss/chunked_test/<field>_l2err/last``). Emits a Markdown table and a
   grouped Plotly bar chart.

2. **Sampling-steps sweep** — per-field deterministic L2 error
   (``loss/test/<field>_l2err/last``) as a function of Euler-step count, read
   from the ``eval_steps<N>/`` siblings of the diffusion run. Emits a CSV-style
   Markdown table and a grouped-bar Plotly chart with one group per step count.

3. **UQ calibration** — per-field Pearson correlation between the ensemble
   per-point standard deviation and the per-point absolute error against
   ground truth (``loss/chunked_test/<field>_uq_corr/last``). Emits a Markdown
   table.

Run from ``recipes/diffusion_uq/``::

    uv run python -m scripts.report \\
        --diffusion-run outputs/abupt_diffusion/30035_2026-05-11_spk1e \\
        --baseline-run  outputs/abupt_diffusion/30403_2026-05-13_wysns \\
        --output-dir    outputs/report
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import plotly.graph_objects as go
import yaml

FIELDS: tuple[str, ...] = (
    "surface_friction",
    "surface_pressure",
    "volume_pressure",
    "volume_velocity",
    "volume_vorticity",
)

_STEPS_DIR_RE = re.compile(r"^eval_steps0*(\d+)$")


def _load_summary(run_dir: Path) -> dict[str, float]:
    """Locate and parse ``tracker/summary.yaml`` (incl. nested ``eval_uq``)."""
    candidates = [
        run_dir / "tracker" / "summary.yaml",
        run_dir / "eval_uq" / "tracker" / "summary.yaml",
    ]
    for path in candidates:
        if path.is_file():
            with path.open() as fh:
                return yaml.safe_load(fh)
    raise FileNotFoundError(f"No summary.yaml under {run_dir}")


def _extract(summary: dict[str, float], template: str) -> dict[str, float]:
    return {field: float(summary[template.format(field=field)]) for field in FIELDS}


def _markdown_table(diff: dict[str, float], base: dict[str, float]) -> str:
    rows = [
        "| Field | Regression baseline | Diffusion (4 steps, 6 draws) | Δ (diff − base) | Relative |",
        "|---|---:|---:|---:|---:|",
    ]
    for field in FIELDS:
        b, d = base[field], diff[field]
        delta = d - b
        rel = delta / b if b else float("nan")
        rows.append(f"| `{field}` | {b:.4f} | {d:.4f} | {delta:+.4f} | {rel:+.1%} |")
    return "\n".join(rows)


def _bar_figure(diff: dict[str, float], base: dict[str, float]) -> go.Figure:
    fields = list(FIELDS)
    fig = go.Figure(
        data=[
            go.Bar(
                name="Regression baseline",
                x=fields,
                y=[base[f] for f in fields],
                marker_color="#4C78A8",
                text=[f"{base[f]:.3f}" for f in fields],
                textposition="outside",
            ),
            go.Bar(
                name="Diffusion (4 steps, 6 draws)",
                x=fields,
                y=[diff[f] for f in fields],
                marker_color="#F58518",
                text=[f"{diff[f]:.3f}" for f in fields],
                textposition="outside",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Per-field L2 error: diffusion ensemble vs regression baseline",
        yaxis_title="L2 error (chunked test, last step)",
        xaxis_title="Field",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def _discover_step_runs(run_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        match = _STEPS_DIR_RE.match(child.name)
        if not match:
            continue
        summary = child / "tracker" / "summary.yaml"
        if summary.is_file():
            out[int(match.group(1))] = child
    return dict(sorted(out.items()))


def _load_step_sweep(run_dir: Path) -> dict[int, dict[str, float]]:
    runs = _discover_step_runs(run_dir)
    if not runs:
        raise FileNotFoundError(f"No eval_steps* siblings under {run_dir}")
    return {steps: _extract(_load_summary(path), "loss/test/{field}_l2err/last") for steps, path in runs.items()}


def _sweep_table(sweep: dict[int, dict[str, float]]) -> str:
    step_counts = list(sweep.keys())
    header_cells = " | ".join(f"{n} step{'s' if n != 1 else ''}" for n in step_counts)
    rows = [
        f"| Field | {header_cells} |",
        "|---|" + "---:|" * len(step_counts),
    ]
    for field in FIELDS:
        cells = " | ".join(f"{sweep[n][field]:.4f}" for n in step_counts)
        rows.append(f"| `{field}` | {cells} |")
    return "\n".join(rows)


def _uq_table(uq_corr: dict[str, float]) -> str:
    rows = [
        "| Field | Pearson r(std, \\|error\\|) |",
        "|---|---:|",
    ]
    for field in FIELDS:
        rows.append(f"| `{field}` | {uq_corr[field]:.3f} |")
    return "\n".join(rows)


def _sweep_figure(sweep: dict[int, dict[str, float]]) -> go.Figure:
    step_counts = list(sweep.keys())
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6"]
    bars = [
        go.Bar(
            name=f"{n} step{'s' if n != 1 else ''}",
            x=list(FIELDS),
            y=[sweep[n][field] for field in FIELDS],
            marker_color=palette[i % len(palette)],
        )
        for i, n in enumerate(step_counts)
    ]
    fig = go.Figure(data=bars)
    fig.update_layout(
        barmode="group",
        title="Sampling-steps sweep: per-field L2 error vs Euler-step count",
        yaxis_title="L2 error (deterministic, single draw)",
        xaxis_title="Field",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    diff_summary = _load_summary(args.diffusion_run)
    base_summary = _load_summary(args.baseline_run)

    diff = _extract(diff_summary, "loss/chunked_test/{field}_uq_mean_l2err/last")
    base = _extract(base_summary, "loss/chunked_test/{field}_l2err/last")
    uq_corr = _extract(diff_summary, "loss/chunked_test/{field}_uq_corr/last")

    sweep = _load_step_sweep(args.diffusion_run)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    table_md = _markdown_table(diff, base)
    sweep_md = _sweep_table(sweep)
    uq_md = _uq_table(uq_corr)
    (args.output_dir / "comparison.md").write_text(table_md + "\n")
    (args.output_dir / "sweep.md").write_text(sweep_md + "\n")
    (args.output_dir / "uq.md").write_text(uq_md + "\n")

    cmp_fig = _bar_figure(diff, base)
    cmp_fig.write_html(args.output_dir / "comparison.html", include_plotlyjs="cdn")
    sweep_fig = _sweep_figure(sweep)
    sweep_fig.write_html(args.output_dir / "sweep.html", include_plotlyjs="cdn")

    for fig, name in [(cmp_fig, "comparison.png"), (sweep_fig, "sweep.png")]:
        try:
            fig.write_image(args.output_dir / name, width=900, height=500, scale=2)
        except Exception as exc:  # pragma: no cover - kaleido optional
            print(f"[warn] PNG export skipped for {name}: {exc}")

    print("# Baseline comparison\n")
    print(table_md)
    print("\n# Sampling-steps sweep\n")
    print(sweep_md)
    print("\n# UQ calibration (Pearson r between std and |error|)\n")
    print(uq_md)
    print(f"\nWrote {args.output_dir}/{{comparison,sweep}}.{{md,html,png}} and uq.md")


if __name__ == "__main__":
    main()
