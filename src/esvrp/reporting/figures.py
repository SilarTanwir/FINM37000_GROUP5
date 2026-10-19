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


SERIES_COLORS = ("#2a78d6", "#eb6834")  # categorical slots 1 and 2, in fixed order


def _new_figure(size: tuple[float, float]):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=size, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    return fig


def _save(fig, title: str, subtitle: str, out_path: Path) -> Path:
    fig.suptitle(title, x=0.06, y=0.985, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.text(0.06, 0.905, subtitle, color=INK_SECONDARY, fontsize=9.5)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.13, wspace=0.25)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    return out_path


def _end_label(ax, x, y, text: str, color: str) -> None:
    """Direct label at the end of a line, in text ink beside a colored marker."""
    ax.plot([x], [y], "o", color=color, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                color=INK_SECONDARY)


def h3_figure(joined: pd.DataFrame, stats: pd.DataFrame, out_path: Path) -> Path:
    """VRP against the gamma-weighted hedged gain per chain (H3, planned output 5).

    `joined` and `stats` come from `analysis.reconcile.reconcile`. Left: every chain, coloured by days to
    expiry, with the 45-degree line. Right: the non-overlapping monthly series over time.
    """
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    from esvrp.analysis.reconcile import monthly_subsample

    fig = _new_figure((11, 4.8))
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.5]})
    j = joined.assign(date=lambda d: pd.to_datetime(d["date"]))

    _style(left, "VRP = RV - SW (annualized variance)", "Hedged gain HG (annualized variance)")
    left.grid(axis="x", color=GRID, linewidth=1)
    if len(j):
        j["bucket"] = _maturity_bucket(j["tau"])
        for bucket, color in MATURITY_COLORS.items():
            b = j[j["bucket"] == bucket]
            if len(b):
                left.scatter(b["vrp"], b["hg"], s=14, color=color, alpha=0.6, edgecolors=SURFACE,
                             linewidths=0.4, label=bucket)
        lo = float(np.nanmin([j["vrp"].min(), j["hg"].min()]))
        hi = float(np.nanmax([j["vrp"].max(), j["hg"].max()]))
        left.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1, linestyle="--", label="HG = VRP")
        left.legend(title="Days to expiry", frameon=False, fontsize=9, title_fontsize=9, loc="lower right",
                    labelcolor=INK_SECONDARY)
    left.axhline(0, color=AXIS, linewidth=1)
    left.axvline(0, color=AXIS, linewidth=1)
    left.set_title("Every chain (overlapping windows)", loc="left", color=INK, fontsize=10)

    _style(right, "", "Annualized variance")
    right.xaxis.set_major_formatter(ConciseDateFormatter(AutoDateLocator()))
    m = monthly_subsample(j) if len(j) else j
    for (col, name), color in zip((("vrp", "VRP"), ("hg", "HG")), SERIES_COLORS, strict=True):
        if len(m):
            right.plot(m["date"], m[col], color=color, linewidth=2, label=name, solid_capstyle="round")
            _end_label(right, m["date"].iloc[-1], m[col].iloc[-1], name, color)
    right.axhline(0, color=AXIS, linewidth=1)
    if len(m):
        right.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower left", ncols=2)
    right.set_title("Monthly, one chain per expiry (non-overlapping)", loc="left", color=INK, fontsize=10)

    subtitle = "No chains with both a premium and a hedged gain."
    if len(stats) and stats.iloc[0].get("n", 0) >= 3:
        d, mo = stats.iloc[0], stats.iloc[-1]
        subtitle = (f"{int(d['n']):,} chains: correlation {d['pearson']:.2f} (95% CI {d['ci_low']:.2f} to "
                    f"{d['ci_high']:.2f}), same sign {d['sign_agree']:.0%}. Monthly ({int(mo['n'])}): "
                    f"correlation {mo.get('pearson', np.nan):.2f}.")
    title = "Variance risk premium against gamma-weighted delta-hedged gains (H3)"
    return _save(fig, title, subtitle, out_path)


def strategy_figure(trades: pd.DataFrame, metrics: pd.DataFrame, out_path: Path) -> Path:
    """Monthly short hedged straddle: cumulative P&L gross and net, and each month's net P&L.

    Both panels are per unit of entry vega, in vol points (planned output 6).
    """
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    fig = _new_figure((11, 4.8))
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.4, 1]})
    t = trades.assign(date=lambda d: pd.to_datetime(d["date"])).sort_values("date")

    _style(left, "", "Cumulative P&L (vol points per unit vega)")
    left.xaxis.set_major_formatter(ConciseDateFormatter(AutoDateLocator()))
    for (col, name), color in zip((("gross_per_vega", "Gross"), ("net_per_vega", "Net of costs")),
                                  SERIES_COLORS, strict=True):
        if len(t):
            cum = t[col].cumsum()
            left.step(t["date"], cum, where="post", color=color, linewidth=2, label=name)
            _end_label(left, t["date"].iloc[-1], cum.iloc[-1], name, color)
    left.axhline(0, color=AXIS, linewidth=1)
    if len(t):
        left.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left", ncols=2)
    left.set_title("Sum of monthly trades", loc="left", color=INK, fontsize=10)

    _style(right, "", "Net P&L (vol points per unit vega)")
    right.xaxis.set_major_formatter(ConciseDateFormatter(AutoDateLocator()))
    if len(t):
        right.bar(t["date"], t["net_per_vega"], width=20, color=SERIES_COLORS[1], edgecolor=SURFACE,
                  linewidth=1)
        for _, row in t.nsmallest(min(3, len(t)), "net_per_vega").iterrows():
            right.annotate(f"{row['date']:%b %Y}: {row['net_per_vega']:.1f}",
                           (row["date"], row["net_per_vega"]), xytext=(7, 0), textcoords="offset points",
                           ha="left", va="center", fontsize=8, color=INK_SECONDARY)
        right.margins(y=0.08)
    right.axhline(0, color=AXIS, linewidth=1)
    right.set_title("Each month, net (worst three labelled)", loc="left", color=INK, fontsize=10)

    subtitle = "No trades."
    if "net_per_vega" in metrics.index and metrics.loc["net_per_vega"].get("n", 0):
        m = metrics.loc["net_per_vega"]
        cvar = next((c for c in metrics.columns if str(c).startswith("cvar_")), None)
        subtitle = (f"{int(m['n'])} trades. Net per unit vega: mean {m['mean']:.2f} vol points, Sharpe "
                    f"{m['sharpe']:.2f}, skew {m['skew']:.2f}, worst month {m['worst']:.2f}"
                    + (f", {cvar.replace('cvar_', 'CVaR ')} {m[cvar]:.2f}." if cvar else "."))
    return _save(fig, "Short delta-hedged ATM straddle, sold monthly at the bid", subtitle, out_path)


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
