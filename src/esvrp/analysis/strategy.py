"""Monthly short delta-hedged straddle backtest and risk metrics.

Implements PLANNING.md §4.7 and §6 stage 7.

Each month, sell one delta-hedged at-the-money straddle on the end-of-month expiry, on the first
snapshot of that month, and hold it to expiry; the next trade starts after it expires, so trades never
overlap. The legs come from the stage 6 hedged gains (daily futures hedge, settlement at expiry), so
the short position earns minus the long gain. Selling at the bid is the mid gain less the half-spread,
which is part of each leg's `cost` along with hedge slippage and fees.

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
TRADE_COLUMNS = ["date", "expiry", "underlying", "strike", "F", "moneyness", "days", "iv_call", "iv_put",
                 "vega", "gross", "cost", "net", "gross_per_vega", "net_per_vega", "margin", "gross_return",
                 "net_return"]


@dataclass
class StrategyResult:
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)  # one row per monthly trade
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)  # rows: series, columns: risk metrics


def monthly_straddles(hedged_gains: pd.DataFrame, hedge_delta: str = "iv_daily",
                      margin_fraction: float = 0.10, cost_multiplier: float = 1.0) -> pd.DataFrame:
    """One short ATM straddle per end-of-month expiry, entered on the first snapshot of its month.

    The strike is the one closest to the forward `F` that has both a call and a put that day. P&L is in
    dollars for one straddle (one call and one put): `gross` at mid, `net` after `cost_multiplier` times
    the legs' costs. `vega` is the straddle's dollar vega per vol point at entry.
    """
    g = hedged_gains[hedged_gains["hedge_delta"] == hedge_delta]
    if g.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    g = g.assign(month=pd.to_datetime(g["date"]).dt.to_period("M"),
                 expiry_month=pd.to_datetime(g["expiry"], utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
                 .dt.to_period("M"))
    g = g[g["month"] == g["expiry_month"]]
    g = g[g["date"] == g.groupby("expiry")["date"].transform("min")]

    legs = g.pivot_table(index=["date", "expiry", "strike"], columns="cp",
                         values=["gain", "cost", "vega", "iv", "F", "moneyness", "underlying", "tau"],
                         aggfunc="first")
    if not {"c", "p"} <= set(legs["gain"].columns):
        return pd.DataFrame(columns=TRADE_COLUMNS)
    legs = legs[legs["gain"].notna().all(axis=1)]  # both a call and a put at that strike
    if legs.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    legs["abs_m"] = legs[("moneyness", "c")].astype(float).abs()
    atm = legs.loc[legs.groupby(level="expiry")["abs_m"].idxmin()]

    t = pd.DataFrame({
        "date": atm.index.get_level_values("date"), "expiry": atm.index.get_level_values("expiry"),
        "underlying": atm[("underlying", "c")].to_numpy(), "strike": atm.index.get_level_values("strike"),
        "F": atm[("F", "c")].to_numpy(float), "moneyness": atm[("moneyness", "c")].to_numpy(float),
        "days": atm[("tau", "c")].to_numpy(float) * 365,
        "iv_call": atm[("iv", "c")].to_numpy(float), "iv_put": atm[("iv", "p")].to_numpy(float),
        "vega": (atm[("vega", "c")] + atm[("vega", "p")]).to_numpy(float) * ES_MULTIPLIER * 0.01,
        "gross": -(atm[("gain", "c")] + atm[("gain", "p")]).to_numpy(float),
        "cost": cost_multiplier * (atm[("cost", "c")] + atm[("cost", "p")]).to_numpy(float),
    })
    t["net"] = t["gross"] - t["cost"]
    t["gross_per_vega"], t["net_per_vega"] = t["gross"] / t["vega"], t["net"] / t["vega"]
    t["margin"] = margin_fraction * 2 * t["F"] * ES_MULTIPLIER
    t["gross_return"], t["net_return"] = t["gross"] / t["margin"], t["net"] / t["margin"]
    return t.sort_values("date").reset_index(drop=True)[TRADE_COLUMNS]


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
                 cost_multiplier: float = 1.0, cvar_level: float = 0.05) -> StrategyResult:
    """Sell one hedged ATM straddle monthly at the bid, charge all costs, report tails."""
    # Decided (§4.7, §15.5): entry vega primary; assumed margin fraction secondary.
    trades = monthly_straddles(hedged_gains, hedge_delta, margin_fraction, cost_multiplier)
    series = {"gross_per_vega": "vol points", "net_per_vega": "vol points", "gross": "dollars",
              "net": "dollars", "gross_return": "return on margin", "net_return": "return on margin"}
    metrics = pd.DataFrame({name: risk_metrics(trades[name], cvar_level=cvar_level) for name in series}).T
    metrics.insert(0, "units", pd.Series(series))
    return StrategyResult(trades=trades, metrics=metrics)
