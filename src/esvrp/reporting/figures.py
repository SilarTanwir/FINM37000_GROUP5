"""Figures for the report.

Implements PLANNING.md §6 (Planned outputs 1-7) and the stage diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Maturity bucket -> line colour. Fixed order, so a bucket keeps its colour when another is absent.
MATURITY_COLORS = {"up to 30 days": "#2a78d6", "31 to 60 days": "#eb6834", "over 60 days": "#1baf7a"}
SURFACE, INK, INK_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"


def _maturity_bucket(tau: pd.Series) -> pd.Series:
    days = tau * 365
    short, mid, long = MATURITY_COLORS
    return pd.Series(np.select([days <= 30, days <= 60], [short, mid], long), index=tau.index)


def _style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)


def forward_gap_figure(gaps: pd.DataFrame, out_path: Path) -> Path:
    """Parity-implied forward minus futures mid, per chain (roadmap 8 diagnostic).

    `gaps` has one row per chain with date, tau and forward_gap (see `pricing.chain.forward_gap_table`).
    Left: daily median gap by maturity bucket. Right: distribution of the gap over all chains.
    Both panels are in index points (one ES tick is 0.25).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    g = gaps.dropna(subset=["forward_gap"]).assign(date=lambda d: pd.to_datetime(d["date"]))
    g["bucket"] = _maturity_bucket(g["tau"])
    fig = Figure(figsize=(11, 4.6), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2.4, 1], "wspace": 0.22})

    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    _style(left, "", "Gap (index points)")
    left.xaxis.set_major_formatter(ConciseDateFormatter(AutoDateLocator()))
    daily = g.groupby(["bucket", "date"])["forward_gap"].median()
    for bucket, color in MATURITY_COLORS.items():
        if bucket in daily.index.get_level_values("bucket"):
            left.plot(daily[bucket].index, daily[bucket].to_numpy(), color=color, linewidth=2, label=bucket,
                      solid_capstyle="round", solid_joinstyle="round")
    left.axhline(0, color=AXIS, linewidth=1)
    if len(g):  # above the plot, so it never covers a line
        left.legend(title="Daily median, by days to expiry", frameon=False, fontsize=9, title_fontsize=10,
                    labelcolor=INK_SECONDARY, loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncols=3,
                    alignment="left", borderaxespad=0.2)

    _style(right, "Gap (index points)", "Chains")
    lo, hi = g["forward_gap"].quantile([0.005, 0.995]) if len(g) else (-1.0, 1.0)
    right.hist(g["forward_gap"].clip(lo, hi), bins=40, color=MUTED, edgecolor=SURFACE, linewidth=1)
    right.axvline(0, color=AXIS, linewidth=1)
    right.set_title("All chains (tails clipped at 0.5%)", loc="left", color=INK, fontsize=10)

    gap = g["forward_gap"]
    summary = "No chains with a parity forward."
    if len(g):
        summary = (f"{len(g):,} chains. Median gap {gap.median():+.2f}, median absolute gap "
                   f"{gap.abs().median():.2f}, {(gap.abs() <= 1).mean():.0%} within 1 point.")
    fig.suptitle("Gap between the forward implied by put-call parity and the futures mid", x=0.06, y=0.985,
                 ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.text(0.06, 0.905, summary, color=INK_SECONDARY, fontsize=9.5)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.76, bottom=0.12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    return out_path


def make_figures(results: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    raise NotImplementedError
