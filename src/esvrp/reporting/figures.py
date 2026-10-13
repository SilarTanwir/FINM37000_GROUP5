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


def _canvas(figsize: tuple[float, float], ncols: int, ratios: list[float] | None = None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, ncols, gridspec_kw={"width_ratios": ratios or [1] * ncols, "wspace": 0.22})
    return fig, axes


def _finish(fig, title: str, summary: str, out_path: Path) -> Path:
    fig.suptitle(title, x=0.06, y=0.985, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.text(0.06, 0.905, summary, color=INK_SECONDARY, fontsize=9.5)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.76, bottom=0.12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    return out_path


def _complete(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rows with every column present, with `date` as a datetime.

    (Assigning a column onto an emptied frame would bring the dropped rows back, so convert in place.)
    """
    d = df.dropna(subset=columns).copy()
    d["date"] = pd.to_datetime(d["date"])
    return d


def _date_axis(ax) -> None:
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    ax.xaxis.set_major_formatter(ConciseDateFormatter(AutoDateLocator()))


def _identity_line(ax, x: pd.Series, y: pd.Series) -> None:
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1)


def sw_vs_rv_figure(vrp: pd.DataFrame, out_path: Path) -> Path:
    """Plot 1 of §6: swap rate vs realized variance over time, and a 45-degree scatter.

    `vrp` has `date`, `sw` and `rv` (from `premium.vrp_table`). Left: daily median of each. Right: one
    point per chain; points below the line have RV < SW, a negative premium.
    """
    d = _complete(vrp, ["sw", "rv"])
    fig, (left, right) = _canvas((11, 4.6), 2, [2.4, 1])
    _style(left, "", "Variance (annualized)")
    _date_axis(left)
    daily = d.groupby("date")[["sw", "rv"]].median()
    left.plot(daily.index, daily["sw"].to_numpy(), color="#2a78d6", linewidth=2, label="Swap rate (SW)")
    left.plot(daily.index, daily["rv"].to_numpy(), color="#eb6834", linewidth=2, label="Realized (RV)")
    if len(d):
        left.legend(title="Daily median", frameon=False, fontsize=9, title_fontsize=10,
                    labelcolor=INK_SECONDARY, loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncols=2,
                    alignment="left", borderaxespad=0.2)
    _style(right, "SW", "RV")
    right.scatter(d["sw"], d["rv"], s=8, color=MUTED, alpha=0.6, linewidths=0)
    if len(d):
        _identity_line(right, d["sw"], d["rv"])
    right.set_title("One point per chain", loc="left", color=INK, fontsize=10)
    summary = "No chains with both SW and RV."
    if len(d):
        summary = (f"{len(d):,} chains. RV is below SW in {(d['rv'] < d['sw']).mean():.0%}; mean SW "
                   f"{d['sw'].mean():.4f}, mean RV {d['rv'].mean():.4f}.")
    return _finish(fig, "Swap rate versus realized variance", summary, out_path)


def vrp_distribution_figure(vrp: pd.DataFrame, out_path: Path, window: int = 21) -> Path:
    """Plot 2 of §6: histogram of the premium RV - SW and its rolling mean over time.

    `vrp` has `date` and `vrp_diff`. Right: daily median premium and its `window`-day rolling mean.
    """
    d = _complete(vrp, ["vrp_diff"])
    fig, (left, right) = _canvas((11, 4.6), 2, [1, 2.4])
    _style(left, "RV - SW", "Chains")
    lo, hi = d["vrp_diff"].quantile([0.005, 0.995]) if len(d) else (-1.0, 1.0)
    left.hist(d["vrp_diff"].clip(lo, hi), bins=40, color=MUTED, edgecolor=SURFACE, linewidth=1)
    left.axvline(0, color=AXIS, linewidth=1)
    if len(d):
        left.axvline(d["vrp_diff"].mean(), color="#eb6834", linewidth=2)
    left.set_title("All chains (tails clipped at 0.5%); orange = mean", loc="left", color=INK, fontsize=10)
    _style(right, "", "RV - SW")
    _date_axis(right)
    daily = d.groupby("date")["vrp_diff"].median()
    right.plot(daily.index, daily.to_numpy(), color=MUTED, linewidth=1, label="Daily median")
    right.plot(daily.index, daily.rolling(window, min_periods=max(2, window // 3)).mean().to_numpy(),
               color="#2a78d6", linewidth=2, label=f"{window}-day rolling mean")
    right.axhline(0, color=AXIS, linewidth=1)
    if len(d):
        right.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower left",
                     bbox_to_anchor=(-0.01, 1.0), ncols=2, alignment="left", borderaxespad=0.2)
    summary = "No chains with a premium."
    if len(d):
        x = d["vrp_diff"]
        summary = (f"{len(x):,} chains. Mean {x.mean():+.4f}, median {x.median():+.4f}, "
                   f"negative in {(x < 0).mean():.0%}. Overlapping observations: not independent.")
    return _finish(fig, "Variance risk premium (RV - SW)", summary, out_path)


def rv_comparison_figure(rv: pd.DataFrame, out_path: Path) -> Path:
    """Plot 3 of §6: daily (settlement) vs 5-minute realized variance.

    `rv` has `date`, `rv_daily` and `rv_5m` (from `realized.rv_table`). Left: daily median of each.
    Right: one point per chain.
    """
    d = _complete(rv, ["rv_daily", "rv_5m"])
    fig, (left, right) = _canvas((11, 4.6), 2, [2.4, 1])
    _style(left, "", "Realized variance (annualized)")
    _date_axis(left)
    daily = d.groupby("date")[["rv_5m", "rv_daily"]].median()
    left.plot(daily.index, daily["rv_5m"].to_numpy(), color="#2a78d6", linewidth=2, label="5-minute")
    left.plot(daily.index, daily["rv_daily"].to_numpy(), color="#eb6834", linewidth=2, label="Daily")
    if len(d):
        left.legend(title="Daily median", frameon=False, fontsize=9, title_fontsize=10,
                    labelcolor=INK_SECONDARY, loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncols=2,
                    alignment="left", borderaxespad=0.2)
    _style(right, "5-minute RV", "Daily RV")
    right.scatter(d["rv_5m"], d["rv_daily"], s=8, color=MUTED, alpha=0.6, linewidths=0)
    if len(d):
        _identity_line(right, d["rv_5m"], d["rv_daily"])
    right.set_title("One point per chain", loc="left", color=INK, fontsize=10)
    summary = "No chains with both daily and 5-minute RV."
    if len(d):
        summary = (f"{len(d):,} chains. Mean 5-minute {d['rv_5m'].mean():.4f}, daily "
                   f"{d['rv_daily'].mean():.4f}; correlation {d['rv_5m'].corr(d['rv_daily']):.2f}.")
    return _finish(fig, "Daily versus 5-minute realized variance", summary, out_path)


def make_figures(results: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    raise NotImplementedError
