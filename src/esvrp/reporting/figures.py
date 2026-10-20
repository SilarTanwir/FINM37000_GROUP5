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


def chain_quality_figure(quality: pd.DataFrame, out_path: Path) -> Path:
    """Chain completeness, quote freshness and synchronization through time (roadmap 20 diagnostic).

    `quality` has one row per chain (see `data.quality.chain_quality`). Top: strikes per chain, daily median
    and minimum. Middle: share of stale quotes per chain, daily median and maximum. Bottom: age of the
    futures observation and median option-futures gap, daily maximum (empty if stage 2 did not record them).
    Dates with at least one failing chain are marked in the top panel.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    from matplotlib.figure import Figure

    q = quality.assign(date=lambda d: pd.to_datetime(d["date"]))
    fig = Figure(figsize=(11, 8.2), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    top, mid, bottom = fig.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0.32})
    day = q.groupby("date")
    blue, orange = MATURITY_COLORS["up to 30 days"], MATURITY_COLORS["31 to 60 days"]

    _style(top, "", "Strikes per chain")
    top.plot(day["n_strikes"].median(), color=blue, linewidth=2, label="median")
    top.plot(day["n_strikes"].min(), color=MUTED, linewidth=1.5, label="minimum")
    failing = q.loc[q["reasons"] != ""].groupby("date").size()
    if len(failing):
        top.scatter(failing.index, np.zeros(len(failing)), marker="|", s=90, color=orange, zorder=3,
                    label="day with a failing chain", clip_on=False)
    top.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left", ncols=3,
               bbox_to_anchor=(0, 1.2))

    _style(mid, "", "Stale quotes (share)")
    mid.plot(day["stale_share"].median(), color=blue, linewidth=2, label="median")
    mid.plot(day["stale_share"].max(), color=orange, linewidth=1.5, label="worst chain")
    mid.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left", ncols=2,
               bbox_to_anchor=(0, 1.2))

    _style(bottom, "", "Seconds")
    if q["F_age_s"].notna().any():
        bottom.plot(day["F_age_s"].max(), color=blue, linewidth=2, label="futures observation age")
        bottom.plot(day["sync_gap_s"].max(), color=orange, linewidth=1.5, label="option-futures gap")
        bottom.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left", ncols=2,
                      bbox_to_anchor=(0, 1.2))
    else:
        bottom.text(0.5, 0.5, "Futures observation time not recorded in this sample", ha="center",
                    transform=bottom.transAxes, color=MUTED, fontsize=10)
    bottom.xaxis.set_major_formatter(ConciseDateFormatter(AutoDateLocator()))

    n_fail, n = int((q["reasons"] != "").sum()), len(q)
    summary = (f"{n:,} chains over {q['date'].nunique():,} days; {n_fail:,} fail at least one rule "
               f"({n_fail / n:.1%})." if n else "No chains.")
    fig.suptitle("Option-chain quality: coverage, freshness and synchronization", x=0.06, y=0.985,
                 ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.text(0.06, 0.95, summary, color=INK_SECONDARY, fontsize=9.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.07)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    return out_path


def make_figures(results: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    raise NotImplementedError
