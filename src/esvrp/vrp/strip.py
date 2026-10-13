"""Variance-swap strip with diagnostics.

Implements PLANNING.md §4.3 (model-free strip; truncation, discretization, jumps, stale quotes).

The synthetic variance swap rate of one chain (one expiry on one day) is the VIX-style discrete sum

    SW = (2/tau) e^{r tau} sum_i (dK_i / K_i^2) Q(K_i) - (1/tau) (F/K0 - 1)^2

where K0 is the highest strike at or below F, Q is the out-of-the-money option price at each strike (put
below K0, call above, average of put and call at K0) and dK_i is half the distance between the
neighbouring strikes (one-sided at the ends). `SW` is in annualized variance units: 0.04 is a 20%
volatility.

Sources. The continuous result is Carr and Wu (2009), Proposition 1, Eq. (5): E^Q[RV] =
(2/(T-t)) * integral of Theta(K)/(B K^2) dK + jump error, with e^{r tau} = 1/B. The discrete sum, the
definitions of K0, dK and Q(K0), and the (F/K0 - 1)^2 correction are the Cboe VIX methodology
(docs/references/Cboe VIX Mathematics Methodology.pdf, "Calculating Volatility", formula (iv)). Carr
and Wu do not use the discrete sum: their implementation interpolates implied volatility onto a fine
grid (the `iv_grid` variant of §4.3, not implemented here; see the roadmap issue).

Differences from the Cboe rules, deliberate or inherited from earlier stages:
- Cboe stops adding strikes outward after two consecutive zero-bid (or zero-ask) strikes. Stage 2
  already drops zero-bid rows, so that information is gone and strikes beyond a gap are kept, with a
  wider dK (see `max_strike_gap`).
- Cboe requires both a put and a call at K0 and cannot compute otherwise. Here a single side is used
  and `k0_one_sided` is set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("mid", "bid", "ask")


@dataclass
class StripResult:
    sw: float
    k0: float
    n_strikes: int
    strike_min: float
    strike_max: float
    truncation_flag: bool
    contributions: pd.Series = field(default_factory=pd.Series)  # per-strike contributions to SW
    max_strike_gap: float = float("nan")  # widest dK between used strikes (discretization diagnostic)
    k0_one_sided: bool = False  # only a call or only a put was available at K0
    correction: float = float("nan")  # the (F/K0 - 1)^2 / tau term that was subtracted


def _empty(truncation: bool = True) -> StripResult:
    nan = float("nan")
    return StripResult(nan, nan, 0, nan, nan, truncation, pd.Series(dtype=float))


def variance_swap_rate(
    chain_slice: pd.DataFrame, F: float, r: float, tau: float, price_col: str = "mid",
    min_range_pct: float = 0.10,
) -> StripResult:
    """VIX-style discrete strip (baseline) with bid/mid/ask variants.

    `chain_slice` is one expiry on one day with columns `strike`, `cp` ("c" or "p") and `price_col`.
    Quotes with a missing or non-positive price are ignored, so one-sided quotes enter the `bid` variant
    but not `mid` or `ask`. Returns NaN (and `n_strikes == 0`) if `F` or `tau` is invalid or no OTM
    option is available. `truncation_flag` is set when the strikes do not span at least
    F(1 - min_range_pct) to F(1 + min_range_pct), which biases SW downward (§4.3, error 1).
    """
    # See also strip.method iv_grid (Carr-Wu implementation) in §4.3.
    if price_col not in PRICE_COLUMNS:
        raise ValueError(f"price_col must be one of {PRICE_COLUMNS}, got {price_col!r}")
    if not (np.isfinite(F) and F > 0 and np.isfinite(tau) and tau > 0 and np.isfinite(r)):
        return _empty()

    q = chain_slice[["strike", "cp", price_col]].rename(columns={price_col: "price"})
    q = q[q["price"].notna() & (q["price"] > 0) & q["strike"].notna()]
    if q.empty or not (q["strike"] <= F).any():
        return _empty()
    k0 = float(q.loc[q["strike"] <= F, "strike"].max())

    # Average if a strike has duplicate quotes, then pivot to one put and one call column per strike.
    px = q.groupby(["strike", "cp"])["price"].mean().unstack("cp").reindex(columns=["p", "c"])
    otm = pd.Series(np.nan, index=px.index)
    below, above = px.index < k0, px.index > k0
    otm[below] = px.loc[below, "p"]
    otm[above] = px.loc[above, "c"]
    at_k0 = px.index == k0
    if at_k0.any():
        otm[at_k0] = px.loc[at_k0, ["p", "c"]].mean(axis=1)  # mean of the available sides
    k0_one_sided = bool(at_k0.any() and px.loc[at_k0, ["p", "c"]].isna().any(axis=None))
    otm = otm.dropna()
    if otm.empty:
        return _empty()

    strikes = otm.index.to_numpy(dtype=float)
    if len(strikes) == 1:
        dk = np.array([np.nan])
    else:
        dk = np.empty_like(strikes)
        dk[1:-1] = (strikes[2:] - strikes[:-2]) / 2
        dk[0], dk[-1] = strikes[1] - strikes[0], strikes[-1] - strikes[-2]
    contrib = (2.0 / tau) * math.exp(r * tau) * dk / strikes**2 * otm.to_numpy()
    correction = (F / k0 - 1.0) ** 2 / tau
    contributions = pd.Series(contrib, index=otm.index, name="contribution")
    sw = float(np.nansum(contrib) - correction) if len(strikes) > 1 else float("nan")

    lo, hi = float(strikes.min()), float(strikes.max())
    truncated = bool(lo > F * (1 - min_range_pct) or hi < F * (1 + min_range_pct))
    return StripResult(
        sw=sw, k0=k0, n_strikes=len(strikes), strike_min=lo, strike_max=hi, truncation_flag=truncated,
        contributions=contributions,
        max_strike_gap=float(np.nanmax(dk)) if len(strikes) > 1 else float("nan"),
        k0_one_sided=k0_one_sided, correction=float(correction),
    )


def strip_table(chain: pd.DataFrame, min_range_pct: float = 0.10) -> pd.DataFrame:
    """Run the strip on every chain (date, expiry) of `chain.parquet`; one row per chain.

    Uses the chain's own `F`, `r` and `tau` (stage 4). Columns: date, expiry, underlying, tau, F,
    snapshot_ts (if in the chain), then `sw_mid`, `sw_bid`, `sw_ask` and the diagnostics of the mid
    strip (k0, n_strikes, strike range, truncation flag, widest strike gap, correction term).
    """
    rows = []
    for (date, expiry), g in chain.groupby(["date", "expiry"], sort=True):
        F, r, tau = float(g["F"].iloc[0]), float(g["r"].iloc[0]), float(g["tau"].iloc[0])
        res = {col: variance_swap_rate(g, F, r, tau, col, min_range_pct) for col in PRICE_COLUMNS}
        mid = res["mid"]
        rows.append({
            "date": date, "expiry": expiry, "underlying": g["underlying"].iloc[0], "tau": tau, "F": F,
            "snapshot_ts": g["snapshot_ts"].iloc[0] if "snapshot_ts" in g else pd.NaT,
            "sw_mid": mid.sw, "sw_bid": res["bid"].sw, "sw_ask": res["ask"].sw, "k0": mid.k0,
            "n_strikes": mid.n_strikes, "strike_min": mid.strike_min, "strike_max": mid.strike_max,
            "truncation_flag": mid.truncation_flag, "max_strike_gap": mid.max_strike_gap,
            "k0_one_sided": mid.k0_one_sided, "correction": mid.correction,
        })
    return pd.DataFrame(rows)
