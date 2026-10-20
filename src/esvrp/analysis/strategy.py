"""Monthly short delta-hedged straddle backtest and risk metrics.

Implements PLANNING.md §4.7 and §6 stage 7.

Each month, sell one delta-hedged at-the-money straddle on the end-of-month expiry, on the first
snapshot of that month, and hold it to expiry; the next trade starts after it expires, so trades never
overlap. The legs come from the stage 6 hedged gains (daily futures hedge, settlement at expiry), so
the short position earns minus the long gain. Selling at the bid is the mid gain less the half-spread,
which is part of each leg's `cost` along with hedge slippage and fees.

The baseline is the at-the-money straddle. As a robustness check the same monthly rule is run on a
strangle (sell an out-of-the-money put and call, 5% from the forward by default): it earns less premium
and is less exposed to a move near the strikes, so it shows whether the result depends on selling the
at-the-money options (`run_structures`). `worst_months` lists the worst trades with the market move
behind each.

Returns (decided, §4.7 and §15.5):
- per unit of entry vega, primary: P&L divided by the straddle's dollar vega per vol point, so about
  implied minus realized volatility in vol points for the seller, comparable across months;
- on an assumed margin, secondary and an assumption: P&L over `margin_fraction` of the two legs'
  futures notional. Drawdown and tail metrics are read on this series.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from esvrp.analysis.inference import mean_with_se
from esvrp.hedging.costs import ES_MULTIPLIER

TZ = "America/Chicago"
STRUCTURES = ("straddle", "strangle")
TRADE_COLUMNS = ["date", "expiry", "underlying", "structure", "strike", "strike_call", "strike_put", "F",
                 "F_T", "move", "moneyness", "moneyness_call", "moneyness_put", "days", "iv_call", "iv_put",
                 "vega", "gross", "cost", "net", "gross_per_vega", "net_per_vega", "margin", "gross_return",
                 "net_return"]
WORST_COLUMNS = ["rank", "date", "expiry", "structure", "net_per_vega", "net", "gross", "cost", "move",
                 "iv_entry", "days"]


@dataclass
class StrategyResult:
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)  # one row per monthly trade
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)  # rows: series, columns: risk metrics
    worst: pd.DataFrame = field(default_factory=pd.DataFrame)  # the worst months, with the move behind each


def _entry_legs(hedged_gains: pd.DataFrame, hedge_delta: str) -> pd.DataFrame:
    """The legs available on each end-of-month expiry's entry date (first snapshot of the expiry's month)."""
    g = hedged_gains[hedged_gains["hedge_delta"] == hedge_delta]
    if g.empty:
        return g
    g = g.assign(month=pd.to_datetime(g["date"]).dt.to_period("M"),
                 expiry_month=pd.to_datetime(g["expiry"], utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
                 .dt.to_period("M"))
    g = g[g["month"] == g["expiry_month"]]
    return g[g["date"] == g.groupby("expiry")["date"].transform("min")]


def _pick_straddle(day: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    """The strike closest to the forward that has both a call and a put."""
    calls, puts = day[day["cp"] == "c"].set_index("strike"), day[day["cp"] == "p"].set_index("strike")
    common = calls.index.intersection(puts.index)
    if common.empty:
        return None
    k = min(common, key=lambda s: abs(float(calls.loc[s, "moneyness"])))
    return calls.loc[k].rename(k), puts.loc[k].rename(k)


def _pick_strangle(day: pd.DataFrame, otm: float, tolerance: float) -> tuple[pd.Series, pd.Series] | None:
    """The out-of-the-money call and put closest to `otm` (a share of F) above and below the forward.

    Targets are log-moneyness ln(1 + otm) for the call and ln(1 - otm) for the put, the same
    coordinates as the chain's `moneyness`. A leg further than `tolerance` (log points) from its target
    is not used, so a sparse chain gives no trade, not a different trade.
    """
    calls = day[(day["cp"] == "c") & (day["moneyness"] > 0)]
    puts = day[(day["cp"] == "p") & (day["moneyness"] < 0)]
    if calls.empty or puts.empty:
        return None
    dc, dp = (calls["moneyness"] - np.log1p(otm)).abs(), (puts["moneyness"] - np.log1p(-otm)).abs()
    if dc.min() > tolerance or dp.min() > tolerance:
        return None
    return calls.loc[dc.idxmin()], puts.loc[dp.idxmin()]


def monthly_trades(hedged_gains: pd.DataFrame, structure: str = "straddle", hedge_delta: str = "iv_daily",
                   margin_fraction: float = 0.10, cost_multiplier: float = 1.0, otm: float = 0.05,
                   tolerance: float = 0.02) -> pd.DataFrame:
    """One short structure per end-of-month expiry, entered on the first snapshot of its month.

    - `straddle`: the strike closest to the forward `F` that has both a call and a put.
    - `strangle`: the out-of-the-money call and put about `otm` (5%) above and below `F`
      (see `_pick_strangle`).

    P&L is in dollars for one structure (one call and one put): `gross` at mid, `net` after
    `cost_multiplier` times the legs' costs. `vega` is the structure's dollar vega per vol point at entry,
    `F_T` the futures settlement at expiry and `move = F_T / F - 1` the market move it had to survive.
    The margin base is two legs' futures notional (an assumption, as before), for both structures.
    """
    if structure not in STRUCTURES:
        raise ValueError(f"structure must be one of {STRUCTURES}, got {structure!r}")
    g = _entry_legs(hedged_gains, hedge_delta)
    if g.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    rows = []
    for (_, _), day in g.groupby(["date", "expiry"], sort=True):
        pair = _pick_straddle(day) if structure == "straddle" else _pick_strangle(day, otm, tolerance)
        if pair is None:
            continue
        c, p = pair
        gross = -(float(c["gain"]) + float(p["gain"]))
        cost = cost_multiplier * (float(c["cost"]) + float(p["cost"]))
        vega = (float(c["vega"]) + float(p["vega"])) * ES_MULTIPLIER * 0.01
        F = float(c["F"])
        F_T = float(c["F_T"]) if "F_T" in c and pd.notna(c["F_T"]) else float("nan")
        same = structure == "straddle"
        k_call = float(c["strike"]) if "strike" in c else float(c.name)  # straddle rows: strike is the index
        k_put = float(p["strike"]) if "strike" in p else float(p.name)
        margin = margin_fraction * 2 * F * ES_MULTIPLIER
        rows.append({
            "date": c["date"], "expiry": c["expiry"], "underlying": c["underlying"], "structure": structure,
            "strike": k_call if same else float("nan"), "strike_call": k_call, "strike_put": k_put,
            "F": F, "F_T": F_T, "move": F_T / F - 1,
            "moneyness": float(c["moneyness"]) if same else float("nan"),
            "moneyness_call": float(c["moneyness"]), "moneyness_put": float(p["moneyness"]),
            "days": float(c["tau"]) * 365, "iv_call": float(c["iv"]), "iv_put": float(p["iv"]), "vega": vega,
            "gross": gross, "cost": cost, "net": gross - cost, "gross_per_vega": gross / vega,
            "net_per_vega": (gross - cost) / vega, "margin": margin, "gross_return": gross / margin,
            "net_return": (gross - cost) / margin,
        })
    if not rows:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)[TRADE_COLUMNS]


def monthly_straddles(hedged_gains: pd.DataFrame, hedge_delta: str = "iv_daily",
                      margin_fraction: float = 0.10, cost_multiplier: float = 1.0) -> pd.DataFrame:
    """The baseline: one short ATM straddle per end-of-month expiry (see `monthly_trades`)."""
    return monthly_trades(hedged_gains, "straddle", hedge_delta, margin_fraction, cost_multiplier)


def worst_months(trades: pd.DataFrame, n: int = 5, col: str = "net_per_vega") -> pd.DataFrame:
    """The `n` worst trades by `col`, with what happened: the market move and the implied vol sold.

    A short straddle or strangle loses when the futures settle far from where they started, more than
    the premium collected, so each row shows `move` (`F_T / F - 1`) next to `iv_entry` (the average
    implied vol of the two legs). `net`, `gross` and `cost` are in dollars for one structure.
    """
    if trades.empty or col not in trades:
        return pd.DataFrame(columns=WORST_COLUMNS)
    w = trades.dropna(subset=[col]).nsmallest(n, col).copy()
    w["iv_entry"] = (w["iv_call"] + w["iv_put"]) / 2
    w.insert(0, "rank", range(1, len(w) + 1))
    return w[WORST_COLUMNS].reset_index(drop=True)


def risk_metrics(x: pd.Series, periods_per_year: int = 12, cvar_level: float = 0.05,
                 hac_lags: int = 6) -> dict[str, float]:
    """Mean, naive and HAC t-statistics, annualized Sharpe, skew, worst periods, drawdown and CVaR.

    `x` is one value per non-overlapping period (a monthly trade); the HAC lag defaults to Carr-Wu's 6
    for non-overlapping monthly series (§4.6). Drawdown is on the running sum (no compounding). CVaR is
    the mean of the worst `cvar_level` share of periods, at least one.
    """
    x = pd.Series(x, dtype=float).dropna()
    n = len(x)
    if n == 0:
        return {"n": 0}
    naive = mean_with_se(x.to_numpy(), "naive")
    hac = mean_with_se(x.to_numpy(), "hac", lags=min(hac_lags, n - 2)) if n > 2 else naive
    std = x.std(ddof=1) if n > 1 else float("nan")
    cum = x.cumsum()
    worst = np.sort(x.to_numpy())
    k = max(1, int(np.floor(cvar_level * n)))
    return {
        "n": n, "mean": float(x.mean()), "std": float(std), "t_naive": naive.t, "t_hac": hac.t,
        "sharpe": float(x.mean() / std * np.sqrt(periods_per_year)) if std and std > 0 else float("nan"),
        "skew": float(x.skew()) if n > 2 else float("nan"), "share_positive": float((x > 0).mean()),
        "worst": float(worst[0]), "second_worst": float(worst[1]) if n > 1 else float("nan"),
        "best": float(worst[-1]), "max_drawdown": float((cum - cum.cummax().clip(lower=0)).min()),
        f"cvar_{cvar_level:.0%}": float(worst[:k].mean()),
    }


def run_strategy(hedged_gains: pd.DataFrame, hedge_delta: str = "iv_daily", margin_fraction: float = 0.10,
                 cost_multiplier: float = 1.0, cvar_level: float = 0.05, structure: str = "straddle",
                 otm: float = 0.05, n_worst: int = 5) -> StrategyResult:
    """Sell one hedged ATM straddle (or strangle) monthly at the bid, charge all costs, report tails."""
    # Decided (§4.7, §15.5): entry vega primary; assumed margin fraction secondary.
    trades = monthly_trades(hedged_gains, structure, hedge_delta, margin_fraction, cost_multiplier, otm)
    series = {"gross_per_vega": "vol points", "net_per_vega": "vol points", "gross": "dollars",
              "net": "dollars", "gross_return": "return on margin", "net_return": "return on margin"}
    metrics = pd.DataFrame({name: risk_metrics(trades[name], cvar_level=cvar_level) for name in series}).T
    metrics.insert(0, "units", pd.Series(series))
    return StrategyResult(trades=trades, metrics=metrics, worst=worst_months(trades, n_worst))


def run_structures(hedged_gains: pd.DataFrame, hedge_delta: str = "iv_daily", margin_fraction: float = 0.10,
                   cost_multiplier: float = 1.0, cvar_level: float = 0.05,
                   otms: tuple[float, ...] = (0.03, 0.05, 0.07)) -> pd.DataFrame:
    """Baseline straddle next to strangles at several distances: the robustness table.

    One row per structure (the straddle, then a strangle at each `otm`), net and gross per unit vega:
    number of trades, mean, HAC t, Sharpe, skew, worst month and CVaR. If the strangles tell the same story
    as the straddle, the result does not hinge on selling the at-the-money options.
    """
    variants = [("straddle", None), *(("strangle", o) for o in otms)]
    rows = {}
    for structure, o in variants:
        res = run_strategy(hedged_gains, hedge_delta, margin_fraction, cost_multiplier, cvar_level, structure,
                           o if o is not None else 0.05)
        label = "straddle (ATM)" if o is None else f"strangle ({o:.0%} OTM)"
        net, gross = res.metrics.loc["net_per_vega"], res.metrics.loc["gross_per_vega"]
        cvar = next((c for c in res.metrics.columns if str(c).startswith("cvar_")), None)
        rows[label] = {
            "n": net.get("n", 0),
            "mean_vega_collected": res.trades["vega"].mean() if len(res.trades) else np.nan,
            "gross_mean": gross.get("mean", np.nan), "net_mean": net.get("mean", np.nan),
            "net_t_hac": net.get("t_hac", np.nan), "net_sharpe": net.get("sharpe", np.nan),
            "net_skew": net.get("skew", np.nan), "net_worst": net.get("worst", np.nan),
            "net_cvar": net.get(cvar, np.nan) if cvar else np.nan,
        }
    return pd.DataFrame(rows).T
