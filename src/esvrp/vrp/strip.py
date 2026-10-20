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

from esvrp.pricing.black76 import implied_vols

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
    n_dropped_iv: int = 0  # quotes removed because their implied vol was outside [min_iv, max_iv]
    k0_gap_pct: float = float("nan")  # (F - K0) / F: how far below F the highest usable strike is
    k0_gap_flag: bool = False  # True if the gap exceeded max_k0_gap_pct and SW was left empty


def _empty(truncation: bool = True, n_dropped_iv: int = 0) -> StripResult:
    nan = float("nan")
    return StripResult(nan, nan, 0, nan, nan, truncation, pd.Series(dtype=float), n_dropped_iv=n_dropped_iv)


def _within_iv_bounds(
    q: pd.DataFrame, F: float, r: float, tau: float, min_iv: float, max_iv: float | None
) -> pd.Series:
    """True for quotes whose implied vol lies in [min_iv, max_iv].

    Uses the chain's `iv` (mid) where there is one; quotes without it (one-sided quotes, or a slice
    without an `iv` column) get their implied vol from the price being used. A price that has no implied
    vol (outside the no-arbitrage bounds or the solver's bracket) is dropped too.
    """
    iv = q["iv"].astype(float) if "iv" in q else pd.Series(np.nan, index=q.index)
    missing = iv.isna()
    if missing.any():
        m = q[missing]
        iv.loc[missing] = implied_vols(m["price"], F, m["strike"], tau, r, m["cp"])
    upper = np.inf if max_iv is None else max_iv
    return (iv >= min_iv) & (iv <= upper)


def variance_swap_rate(
    chain_slice: pd.DataFrame, F: float, r: float, tau: float, price_col: str = "mid",
    min_range_pct: float = 0.10, min_iv: float = 0.01, max_iv: float | None = 1.0,
    max_k0_gap_pct: float | None = 0.01,
) -> StripResult:
    """VIX-style discrete strip (baseline) with bid/mid/ask variants.

    `chain_slice` is one expiry on one day with columns `strike`, `cp` ("c" or "p") and `price_col`.
    Quotes with a missing or non-positive price are ignored, so one-sided quotes enter the `bid` variant
    but not `mid` or `ask`. Quotes with an implied vol outside [`min_iv`, `max_iv`] (default 1% to 100%,
    as in Bakshi-Kapadia) are dropped before the strip is built, because a single junk quote far from the
    money can dominate the sum (`n_dropped_iv` counts them); `max_iv=None` turns the cap off.

    K0 is the highest usable strike at or below F. If the quotes around the money are missing, K0 falls
    far below F and the correction `(F/K0 - 1)^2 / tau` outweighs the sum, which can make SW negative
    (impossible for a variance). So when `(F - K0) / F` (`k0_gap_pct`) exceeds `max_k0_gap_pct` (default
    1%), SW is left empty (NaN) and `k0_gap_flag` is set; `None` turns the guard off. Returns
    NaN (and `n_strikes == 0`) if `F` or `tau` is invalid or no OTM option is available.
    `truncation_flag` is set when the strikes do not span at least F(1 - min_range_pct) to
    F(1 + min_range_pct), which biases SW downward (§4.3, error 1).
    """
    # See also strip.method iv_grid (Carr-Wu implementation) in §4.3.
    if price_col not in PRICE_COLUMNS:
        raise ValueError(f"price_col must be one of {PRICE_COLUMNS}, got {price_col!r}")
    if not (np.isfinite(F) and F > 0 and np.isfinite(tau) and tau > 0 and np.isfinite(r)):
        return _empty()

    cols = ["strike", "cp", price_col, *(["iv"] if "iv" in chain_slice else [])]
    q = chain_slice[cols].rename(columns={price_col: "price"})
    q = q[q["price"].notna() & (q["price"] > 0) & q["strike"].notna()]
    n_dropped_iv = 0
    if len(q):
        ok = _within_iv_bounds(q, F, r, tau, min_iv, max_iv)
        n_dropped_iv = int((~ok).sum())
        q = q[ok]
    if q.empty or not (q["strike"] <= F).any():
        return _empty(n_dropped_iv=n_dropped_iv)
    k0 = float(q.loc[q["strike"] <= F, "strike"].max())
    k0_gap = (F - k0) / F

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
        return _empty(n_dropped_iv=n_dropped_iv)

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
    k0_gap_flag = bool(max_k0_gap_pct is not None and k0_gap > max_k0_gap_pct)
    if k0_gap_flag:
        sw = float("nan")

    lo, hi = float(strikes.min()), float(strikes.max())
    truncated = bool(lo > F * (1 - min_range_pct) or hi < F * (1 + min_range_pct))
    return StripResult(
        sw=sw, k0=k0, n_strikes=len(strikes), strike_min=lo, strike_max=hi, truncation_flag=truncated,
        contributions=contributions,
        max_strike_gap=float(np.nanmax(dk)) if len(strikes) > 1 else float("nan"),
        k0_one_sided=k0_one_sided, correction=float(correction), n_dropped_iv=n_dropped_iv,
        k0_gap_pct=float(k0_gap), k0_gap_flag=k0_gap_flag,
    )


def strip_table(
    chain: pd.DataFrame, min_range_pct: float = 0.10, min_iv: float = 0.01, max_iv: float | None = 1.0,
    max_k0_gap_pct: float | None = 0.01,
) -> pd.DataFrame:
    """Run the strip on every chain (date, expiry) of `chain.parquet`; one row per chain.

    Uses the chain's own `F`, `r` and `tau` (stage 4). Columns: date, expiry, underlying, tau, F,
    snapshot_ts (if in the chain), then `sw_mid`, `sw_bid`, `sw_ask` and the diagnostics of the mid
    strip (k0, n_strikes, strike range, truncation flag, widest strike gap, correction term, `k0_gap_pct`
    and `k0_gap_flag`).
    """
    rows = []
    for (date, expiry), g in chain.groupby(["date", "expiry"], sort=True):
        F, r, tau = float(g["F"].iloc[0]), float(g["r"].iloc[0]), float(g["tau"].iloc[0])
        res = {col: variance_swap_rate(g, F, r, tau, col, min_range_pct, min_iv, max_iv, max_k0_gap_pct)
               for col in PRICE_COLUMNS}
        mid = res["mid"]
        rows.append({
            "date": date, "expiry": expiry, "underlying": g["underlying"].iloc[0], "tau": tau, "F": F,
            "snapshot_ts": g["snapshot_ts"].iloc[0] if "snapshot_ts" in g else pd.NaT,
            "sw_mid": mid.sw, "sw_bid": res["bid"].sw, "sw_ask": res["ask"].sw, "k0": mid.k0,
            "n_strikes": mid.n_strikes, "strike_min": mid.strike_min, "strike_max": mid.strike_max,
            "truncation_flag": mid.truncation_flag, "max_strike_gap": mid.max_strike_gap,
            "k0_one_sided": mid.k0_one_sided, "correction": mid.correction, "n_dropped_iv": mid.n_dropped_iv,
            "k0_gap_pct": mid.k0_gap_pct, "k0_gap_flag": mid.k0_gap_flag,
        })
    return pd.DataFrame(rows)
