"""Monthly short delta-hedged straddle backtest and risk metrics.

Implements PLANNING.md §4.7 and §6 stage 7.

Each month, sell one delta-hedged at-the-money straddle on the end-of-month expiry, on the first
snapshot of that month (or, if no straddle can be formed that day, the next snapshot with one, up to five
business days later), and hold it to expiry; the next trade starts after it expires, so trades never
overlap. The legs come from the stage 6 hedged gains (daily futures hedge, settlement at expiry), so
the short position earns minus the long gain. Selling at the bid is the mid gain less the half-spread,
which is part of each leg's `cost` along with hedge slippage and fees.

The baseline is the at-the-money straddle. As a robustness check the same monthly rule is run on a
strangle (sell an out-of-the-money put and call, 5% from the forward by default): it earns less premium
and is less exposed to a move near the strikes, so it shows whether the result depends on selling the
at-the-money options (`run_structures`). `worst_months` lists the worst trades with the market move
behind each.

Known bias: stage 6 hedges each option on its own, so a straddle or strangle is charged slippage and fees
on a futures position for each leg, even though a trader would hedge only the net delta (the legs' deltas
partly cancel). The gross P&L is unaffected (futures P&L is additive); the costs, and so the net P&L, are
overstated. Fixing it needs the legs' daily delta paths from stage 6; see #57 (net-delta hedging)
and #41 (slippage on the change in delta).

Returns (decided, §4.7 and §15.5):
- per unit of entry vega, primary: P&L divided by the straddle's dollar vega per vol point, so about
  implied minus realized volatility in vol points for the seller, comparable across months;
- on an assumed margin, secondary and an assumption: P&L over `margin_fraction` of the two legs'
  futures notional. Drawdown and tail metrics are read on this series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from esvrp.analysis.inference import mean_with_se
from esvrp.hedging.costs import ES_MULTIPLIER

log = logging.getLogger(__name__)

TZ = "America/Chicago"
STRUCTURES = ("straddle", "strangle")
MAX_LOOKAHEAD_DAYS = 5  # business days after the month's first snapshot that an entry may be delayed
SKIPPED_COLUMNS = ["expiry", "first_snapshot", "snapshots_tried", "reason"]
TRADE_COLUMNS = ["date", "entry_delay_days", "expiry", "underlying", "structure", "strike", "strike_call",
                 "strike_put", "F", "F_T", "move", "moneyness", "moneyness_call", "moneyness_put", "days",
                 "iv_call", "iv_put", "vega", "gross", "cost", "net", "gross_per_vega", "net_per_vega",
                 "margin", "gross_return", "net_return"]
WORST_COLUMNS = ["rank", "date", "expiry", "structure", "net_per_vega", "net", "gross", "cost", "move",
                 "iv_entry", "days"]


@dataclass
class StrategyResult:
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)  # one row per monthly trade
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)  # rows: series, columns: risk metrics
    worst: pd.DataFrame = field(default_factory=pd.DataFrame)  # the worst months, with the move behind each
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)  # months with an expiry but no trade, and why


def _month_legs(hedged_gains: pd.DataFrame, hedge_delta: str) -> pd.DataFrame:
    """Every leg whose snapshot is in the expiry's own month (the days a monthly trade may be entered)."""
    g = hedged_gains[hedged_gains["hedge_delta"] == hedge_delta]
    if g.empty:
        return g
    g = g.assign(month=pd.to_datetime(g["date"]).dt.to_period("M"),
                 expiry_month=pd.to_datetime(g["expiry"], utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
                 .dt.to_period("M"))
    return g[g["month"] == g["expiry_month"]]


def _business_days(start: object, end: object) -> int:
    """Business days from `start` to `end` (weekends only; exchange holidays are not removed)."""
    first, last = (np.datetime64(pd.Timestamp(x).date()) for x in (start, end))
    return int(np.busday_count(first, last))


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
    coordinates as the chain's `moneyness`. `tolerance` is in the same units, ln(K/F): 0.02 means a leg
    whose log-moneyness is more than 0.02 (about 2% of F) from its target is not used. For a 5% strangle
    the call target is ln(1.05) = 0.0488, so calls with ln(K/F) between 0.029 and 0.069 are allowed.
    A sparse chain therefore gives no trade, not a different trade.
    """
    calls = day[(day["cp"] == "c") & (day["moneyness"] > 0)]
    puts = day[(day["cp"] == "p") & (day["moneyness"] < 0)]
    if calls.empty or puts.empty:
        return None
    dc, dp = (calls["moneyness"] - np.log1p(otm)).abs(), (puts["moneyness"] - np.log1p(-otm)).abs()
    if dc.min() > tolerance or dp.min() > tolerance:
        return None
    return calls.loc[dc.idxmin()], puts.loc[dp.idxmin()]


def monthly_trades_log(hedged_gains: pd.DataFrame, structure: str = "straddle", hedge_delta: str = "iv_daily",
                       margin_fraction: float = 0.10, cost_multiplier: float = 1.0, otm: float = 0.05,
                       tolerance: float = 0.02, max_lookahead_days: int = MAX_LOOKAHEAD_DAYS,
                       expected_expiries: object = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One short structure per end-of-month expiry, and a log of the expiries that got no trade.

    The trade is entered on the first snapshot of the expiry's month. If the structure cannot be formed
    that day (for example no strike has both a two-sided call and put, as in 2024-06, 2024-07, 2025-11
    and 2025-12 of the first real run, #56), the next snapshots are tried in order, up to
    `max_lookahead_days` business days after the first one (default 5, which still leaves about three
    weeks to expiry). The trade records `entry_delay_days`, the business days between the first snapshot
    and the one used (0 for the first), so a delayed entry is visible.

    - `straddle`: the strike closest to the forward `F` that has both a call and a put.
    - `strangle`: the out-of-the-money call and put about `otm` (5% of F) above and below `F`. `tolerance`
      is in ln(K/F) units (0.02, about 2% of F): a leg further than that from its target gives no trade
      (see `_pick_strangle`).

    P&L is in dollars for one structure (one call and one put): `gross` at mid, `net` after
    `cost_multiplier` times the legs' costs. `vega` is the structure's dollar vega per vol point at entry,
    `F_T` the futures settlement at expiry and `move = F_T / F - 1` the market move it had to survive.
    The margin base is two legs' futures notional (an assumption, as before), for both structures.

    Returns `(trades, skipped)`. `skipped` has one row per expiry with snapshots in its month but no
    trade (`reason`: `no_strike_with_both_legs` or `no_otm_legs_within_tolerance`, with the number of
    snapshots tried). Pass `expected_expiries` (for example every expiry in the sample) to also list those
    with no snapshot at all in their month (`no_entries_in_month`), so every month with an expiry either
    has a trade or a logged reason.
    """
    if structure not in STRUCTURES:
        raise ValueError(f"structure must be one of {STRUCTURES}, got {structure!r}")
    if max_lookahead_days < 0:
        raise ValueError("max_lookahead_days must be non-negative")
    g = _month_legs(hedged_gains, hedge_delta)
    rows, skipped = [], []
    reason = "no_strike_with_both_legs" if structure == "straddle" else "no_otm_legs_within_tolerance"
    seen = set()
    for expiry, legs in (g.groupby("expiry", sort=True) if len(g) else []):
        seen.add(pd.Timestamp(expiry))
        dates = sorted(legs["date"].unique())
        first = dates[0]
        tried = 0
        pair, chosen = None, None
        for d in dates:
            delay = _business_days(first, d)
            if delay > max_lookahead_days:
                break
            tried += 1
            day = legs[legs["date"] == d]
            pair = _pick_straddle(day) if structure == "straddle" else _pick_strangle(day, otm, tolerance)
            if pair is not None:
                chosen = (d, delay)
                break
        if pair is None:
            skipped.append({"expiry": expiry, "first_snapshot": first, "snapshots_tried": tried,
                            "reason": reason})
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
            "date": c["date"], "entry_delay_days": chosen[1], "expiry": c["expiry"],
            "underlying": c["underlying"], "structure": structure,
            "strike": k_call if same else float("nan"), "strike_call": k_call, "strike_put": k_put,
            "F": F, "F_T": F_T, "move": F_T / F - 1,
            "moneyness": float(c["moneyness"]) if same else float("nan"),
            "moneyness_call": float(c["moneyness"]), "moneyness_put": float(p["moneyness"]),
            "days": float(c["tau"]) * 365, "iv_call": float(c["iv"]), "iv_put": float(p["iv"]), "vega": vega,
            "gross": gross, "cost": cost, "net": gross - cost, "gross_per_vega": gross / vega,
            "net_per_vega": (gross - cost) / vega, "margin": margin, "gross_return": gross / margin,
            "net_return": (gross - cost) / margin,
        })
    for expiry in (expected_expiries if expected_expiries is not None else []):
        if pd.Timestamp(expiry) not in seen:
            skipped.append({"expiry": expiry, "first_snapshot": pd.NaT, "snapshots_tried": 0,
                            "reason": "no_entries_in_month"})
    trades = (pd.DataFrame(rows).sort_values("date").reset_index(drop=True)[TRADE_COLUMNS] if rows
              else pd.DataFrame(columns=TRADE_COLUMNS))
    log_df = (pd.DataFrame(skipped, columns=SKIPPED_COLUMNS).sort_values("expiry").reset_index(drop=True)
              if skipped else pd.DataFrame(columns=SKIPPED_COLUMNS))
    delayed = int((trades["entry_delay_days"] > 0).sum()) if len(trades) else 0
    if delayed or len(log_df):
        log.info("Strategy (%s): %d trades, %d entered on a later snapshot, %d months without a trade",
                 structure, len(trades), delayed, len(log_df))
    return trades, log_df


def monthly_trades(hedged_gains: pd.DataFrame, structure: str = "straddle", hedge_delta: str = "iv_daily",
                   margin_fraction: float = 0.10, cost_multiplier: float = 1.0, otm: float = 0.05,
                   tolerance: float = 0.02, max_lookahead_days: int = MAX_LOOKAHEAD_DAYS) -> pd.DataFrame:
    """The trades of `monthly_trades_log` without the log of skipped months."""
    return monthly_trades_log(hedged_gains, structure, hedge_delta, margin_fraction, cost_multiplier, otm,
                              tolerance, max_lookahead_days)[0]


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
                 otm: float = 0.05, n_worst: int = 5, max_lookahead_days: int = MAX_LOOKAHEAD_DAYS,
                 expected_expiries: object = None) -> StrategyResult:
    """Sell one hedged ATM straddle (or strangle) monthly at the bid, charge all costs, report tails.

    An entry that cannot be formed on a month's first snapshot is tried on the next snapshots, up to
    `max_lookahead_days` business days later (`entry_delay_days` on the trade). `skipped` lists the
    months with an expiry but no trade; see `monthly_trades_log`.
    """
    # Decided (§4.7, §15.5): entry vega primary; assumed margin fraction secondary.
    trades, skipped = monthly_trades_log(
        hedged_gains, structure, hedge_delta, margin_fraction, cost_multiplier, otm,
        max_lookahead_days=max_lookahead_days, expected_expiries=expected_expiries)
    series = {"gross_per_vega": "vol points", "net_per_vega": "vol points", "gross": "dollars",
              "net": "dollars", "gross_return": "return on margin", "net_return": "return on margin"}
    metrics = pd.DataFrame({name: risk_metrics(trades[name], cvar_level=cvar_level) for name in series}).T
    metrics.insert(0, "units", pd.Series(series))
    return StrategyResult(trades=trades, metrics=metrics, worst=worst_months(trades, n_worst),
                          skipped=skipped)


def run_structures(hedged_gains: pd.DataFrame, hedge_delta: str = "iv_daily", margin_fraction: float = 0.10,
                   cost_multiplier: float = 1.0, cvar_level: float = 0.05,
                   otms: tuple[float, ...] = (0.03, 0.05, 0.07),
                   max_lookahead_days: int = MAX_LOOKAHEAD_DAYS) -> pd.DataFrame:
    """Baseline straddle next to strangles at several distances: the robustness table.

    One row per structure (the straddle, then a strangle at each `otm`), net and gross per unit vega:
    number of trades (and how many were entered on a later snapshot, `n_delayed`), mean, HAC t, Sharpe,
    skew, worst month and CVaR. If the strangles tell the same story
    as the straddle, the result does not hinge on selling the at-the-money options.

    `cost_per_vega` (gross minus net, per unit of vega) and `cost_usd` show how much of each structure's
    result is costs: a wide strangle sells much less vega but pays the same fixed fees and per-rebalance
    slippage, so its net per-vega result is penalized mostly through costs, not through risk.
    """
    variants = [("straddle", None), *(("strangle", o) for o in otms)]
    rows = {}
    for structure, o in variants:
        res = run_strategy(hedged_gains, hedge_delta, margin_fraction, cost_multiplier, cvar_level, structure,
                           o if o is not None else 0.05, max_lookahead_days=max_lookahead_days)
        label = "straddle (ATM)" if o is None else f"strangle ({o:.0%} OTM)"
        net, gross = res.metrics.loc["net_per_vega"], res.metrics.loc["gross_per_vega"]
        cvar = next((c for c in res.metrics.columns if str(c).startswith("cvar_")), None)
        tr = res.trades
        rows[label] = {
            "n": net.get("n", 0),
            "n_delayed": int((tr["entry_delay_days"] > 0).sum()) if len(tr) else 0,
            "mean_vega_collected": tr["vega"].mean() if len(tr) else np.nan,
            "gross_mean": gross.get("mean", np.nan), "net_mean": net.get("mean", np.nan),
            "cost_per_vega": (tr["cost"] / tr["vega"]).mean() if len(tr) else np.nan,
            "cost_usd": tr["cost"].mean() if len(tr) else np.nan,
            "net_t_hac": net.get("t_hac", np.nan), "net_sharpe": net.get("sharpe", np.nan),
            "net_skew": net.get("skew", np.nan), "net_worst": net.get("worst", np.nan),
            "net_cvar": net.get(cvar, np.nan) if cvar else np.nan,
        }
    return pd.DataFrame(rows).T


def margin_sensitivity(trades: pd.DataFrame, fractions: tuple[float, ...] = (0.06, 0.08, 0.10),
                       cvar_level: float = 0.05) -> pd.DataFrame:
    """How much of the margin the worst month, the CVaR and the maximum drawdown use, per margin fraction.

    The margin base of a trade is `fraction x 2 legs x F x $50` (an assumption, §4.7). The dollar P&L does
    not depend on it. With one fraction for every trade, the return on margin is the dollar P&L divided by
    a number that scales with F, so a different fraction rescales the whole series by 1/fraction: the mean
    return, the t-statistics, the Sharpe ratio and the skew carry no new information, and they are left
    out. What a fraction changes is how large the losses are *relative to the margin*, and that is what
    this table reports: `worst_pct_margin`, `cvar_pct_margin` and `max_drawdown_pct_margin` (-0.5 means
    half the margin was lost, where a margin call becomes a real risk), next to the same losses in dollars
    and the average margin base `margin_usd` per structure.
    """
    dollars = risk_metrics(trades["net"], cvar_level=cvar_level) if len(trades) else {}
    cvar_key = next((k for k in dollars if str(k).startswith("cvar_")), None)
    rows = {}
    for f in fractions:
        if trades.empty:
            rows[f"{f:.0%}"] = {"fraction": f, "n": 0}
            continue
        margin = f * 2 * trades["F"] * ES_MULTIPLIER
        pct = risk_metrics(trades["net"] / margin, cvar_level=cvar_level)
        rows[f"{f:.0%}"] = {
            "fraction": f, "n": pct["n"], "margin_usd": float(margin.mean()),
            "worst_usd": dollars["worst"], "cvar_usd": dollars[cvar_key] if cvar_key else float("nan"),
            "max_drawdown_usd": dollars["max_drawdown"],
            "worst_pct_margin": pct["worst"], "cvar_pct_margin": pct[cvar_key] if cvar_key else float("nan"),
            "max_drawdown_pct_margin": pct["max_drawdown"],
        }
    return pd.DataFrame(rows).T
