"""Generate the three main result charts from the recorded runs.

Every value comes from the saved summary of the main experiment
(benchmarks/results/real-main-seed1234/summary.json); nothing is
typed in by hand. Each chart is a plain SVG bar chart with one bar
per configuration and a whisker showing the range across the 3
repeats behind the mean. Charts are written to docs/charts/.

Usage, from the repository root:

    python benchmarks/make_results_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "real-main-seed1234"
OUT_DIR = ROOT / "docs" / "charts"

WIDTH = 720
HEIGHT = 420
MARGIN_LEFT = 70
MARGIN_RIGHT = 20
MARGIN_TOP = 50
MARGIN_BOTTOM = 70

COLORS = {"short": "#2563eb", "mixed": "#d97706", "serial": "#6b7280"}


def bar_chart(
    title: str,
    y_label: str,
    groups: list[tuple[str, list[tuple[str, float, float, float, str]]]],
    path: Path,
) -> None:
    """Draw grouped bars; each bar is (label, mean, min, max, color key)."""
    plot_w = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    y_max = 0.0
    for _, bars in groups:
        for _, _, _, hi, _ in bars:
            y_max = max(y_max, hi)
    y_max *= 1.15
    if y_max == 0:
        y_max = 1.0

    def y(value: float) -> float:
        return MARGIN_TOP + plot_h - (value / y_max) * plot_h

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{WIDTH / 2}" y="28" text-anchor="middle" '
            f'font-family="sans-serif" font-size="16" font-weight="bold">'
            f"{title}</text>"
        ),
        (
            f'<text x="18" y="{HEIGHT / 2}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" '
            f'transform="rotate(-90 18 {HEIGHT / 2})">{y_label}</text>'
        ),
    ]
    # Gridlines and y tick labels, 4 divisions.
    for i in range(5):
        value = y_max * i / 4
        yy = y(value)
        parts.append(
            f'<line x1="{MARGIN_LEFT}" y1="{yy:.1f}" x2="{WIDTH - MARGIN_RIGHT}" '
            f'y2="{yy:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{MARGIN_LEFT - 6}" y="{yy + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{value:.1f}</text>'
        )
    group_w = plot_w / len(groups)
    for gi, (group_label, bars) in enumerate(groups):
        bar_w = min(46.0, group_w / (len(bars) + 1))
        total_w = bar_w * len(bars) + 6 * (len(bars) - 1)
        start_x = MARGIN_LEFT + gi * group_w + (group_w - total_w) / 2
        for bi, (label, mean, lo, hi, color_key) in enumerate(bars):
            x = start_x + bi * (bar_w + 6)
            parts.append(
                f'<rect x="{x:.1f}" y="{y(mean):.1f}" width="{bar_w:.1f}" '
                f'height="{(y(0) - y(mean)):.1f}" fill="{COLORS[color_key]}"/>'
            )
            cx = x + bar_w / 2
            parts.append(
                f'<line x1="{cx:.1f}" y1="{y(hi):.1f}" x2="{cx:.1f}" '
                f'y2="{y(lo):.1f}" stroke="#111827" stroke-width="1.5"/>'
            )
            parts.append(
                f'<line x1="{cx - 5:.1f}" y1="{y(hi):.1f}" x2="{cx + 5:.1f}" '
                f'y2="{y(hi):.1f}" stroke="#111827" stroke-width="1.5"/>'
            )
            parts.append(
                f'<line x1="{cx - 5:.1f}" y1="{y(lo):.1f}" x2="{cx + 5:.1f}" '
                f'y2="{y(lo):.1f}" stroke="#111827" stroke-width="1.5"/>'
            )
            parts.append(
                f'<text x="{cx:.1f}" y="{HEIGHT - MARGIN_BOTTOM + 18}" '
                f'text-anchor="middle" font-family="sans-serif" '
                f'font-size="11">{label}</text>'
            )
        parts.append(
            f'<text x="{MARGIN_LEFT + gi * group_w + group_w / 2:.1f}" '
            f'y="{HEIGHT - MARGIN_BOTTOM + 40}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" font-weight="bold">'
            f"{group_label}</text>"
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def series(
    summary: dict[str, Any], workload: str, metric: str
) -> list[tuple[str, float, float, float, str]]:
    aggregated = summary["aggregated"]
    bars: list[tuple[str, float, float, float, str]] = []
    serial = aggregated[f"{workload}|serial|1"][metric]
    bars.append(
        ("serial", serial["mean"], serial["min"], serial["max"], "serial")
    )
    for concurrency in (1, 2, 4, 8):
        entry = aggregated[f"{workload}|continuous|{concurrency}"][metric]
        bars.append(
            (
                f"c{concurrency}",
                entry["mean"],
                entry["min"],
                entry["max"],
                workload,
            )
        )
    return bars


def main() -> int:
    summary = json.loads(
        (ROOT / "benchmarks" / "results" / RUN_ID / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    charts = [
        (
            "throughput.svg",
            "Output throughput by strategy and concurrency",
            "output tokens per second",
            "output_tokens_per_second",
        ),
        (
            "ttft.svg",
            "Mean time to first token by strategy and concurrency",
            "mean TTFT, seconds",
            "mean_ttft",
        ),
        (
            "latency.svg",
            "Mean end to end latency by strategy and concurrency",
            "mean latency, seconds",
            "mean_latency",
        ),
    ]
    for filename, title, y_label, metric in charts:
        groups = [
            (workload, series(summary, workload, metric))
            for workload in ("short", "mixed")
        ]
        bar_chart(title, y_label, groups, OUT_DIR / filename)
        print(f"wrote docs/charts/{filename}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
