"""Delta-hedged gain engine.

Implements PLANNING.md §4.5.

One option is bought at t_0 and held to expiry T; at each rebalance date t_0 < ... < t_{N-1} the
position is hedged by shorting Delta_n futures. The futures path F_0, ..., F_N runs from entry to
expiry, so F_N is the futures price at expiry and the option pays its intrinsic value there. Gains are
from the long side; a seller earns -Pi.

Hedge delta variants (§4.5):
- `iv_fixed`: Black-76 delta at the entry implied vol `option["iv"]` throughout.
- `iv_daily`: delta at the option's own implied vol on each rebalance date (`hedge_vol`, from the chain).
- `rv`: delta at a trailing realized-vol estimate on each rebalance date (`hedge_vol`, e.g.
  `trailing_vol`), closest to Bakshi-Kapadia's GARCH/rolling-volatility delta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from esvrp.hedging.costs import ES_MULTIPLIER
from esvrp.pricing.black76 import greeks

SECONDS_PER_YEAR = 365 * 24 * 3600  # ACT/365, as in the chain's tau


@dataclass
class HedgeResult:
    gain_dollars: float
    gain_over_underlying: float
    gain_over_option: float
    n_hedges: int


def delta_hedged_gain(
    option: pd.Series,
    underlying_path: pd.Series,
    rate: float,
    hedge_delta: Literal["iv_daily", "iv_fixed", "rv"],
    entry_price: Literal["mid", "ask", "bid"],
    hedge_vol: pd.Series | None = None,
) -> HedgeResult:
    """Pi = O_T - e^{r tau} O_0 - sum_n Delta_n (F_{n+1} - F_n).

    `option` needs `strike`, `cp`, `tau` (years to expiry at entry), `iv` and the `entry_price` column.
    `underlying_path` is F_0, ..., F_N. With a DatetimeIndex, time to expiry at each rebalance is
    `tau` minus the elapsed time (ACT/365); otherwise the points are taken as equally spaced from entry
    to expiry. `hedge_vol` gives the volatility for delta at each of F_0, ..., F_{N-1} and is required
    for `iv_daily` and `rv` (matched by date when both are dated, else by position); missing values
    carry the last known volatility forward, starting from the entry implied vol.

    Use `entry_price="mid"` together with `apply_costs`, or `"ask"`/`"bid"` without the option
    half-spread there, so the spread is not charged twice.
    """
    F = underlying_path.to_numpy(dtype=float)
    n = len(F) - 1
    if n < 1:
        raise ValueError("underlying_path needs at least an entry and an expiry point")
    K, cp, tau0, iv0 = float(option["strike"]), option["cp"], float(option["tau"]), float(option["iv"])
    O0 = float(option[entry_price])

    taus = _taus_to_expiry(underlying_path.index, tau0, n)
    if isinstance(hedge_vol, pd.Series) and isinstance(underlying_path.index, pd.DatetimeIndex):
        hedge_vol = hedge_vol.reindex(underlying_path.index[:n])
    sigmas = _hedge_sigmas(hedge_delta, hedge_vol, iv0, n)

    deltas = np.array([greeks(F[i], K, taus[i], rate, sigmas[i], cp)["delta"] for i in range(n)])
    payoff = max(F[-1] - K, 0.0) if cp == "c" else max(K - F[-1], 0.0)
    pi = payoff - math.exp(rate * tau0) * O0 - float(np.sum(deltas * np.diff(F)))

    return HedgeResult(
        gain_dollars=pi * ES_MULTIPLIER,
        gain_over_underlying=pi / F[0],
        gain_over_option=pi / O0 if O0 > 0 else float("nan"),
        n_hedges=n,
    )


def trailing_vol(prices: pd.Series, window: int = 21, periods_per_year: int = 252) -> pd.Series:
    """Annualized rolling sample std of log returns up to and including each date (no look-ahead).

    Bakshi-Kapadia's rolling-volatility alternative to a GARCH hedge vol; feed it as `hedge_vol`
    for `hedge_delta="rv"`.
    """
    log_ret = np.log(prices.astype(float)).diff()
    return log_ret.rolling(window).std() * math.sqrt(periods_per_year)


def _taus_to_expiry(index: pd.Index, tau0: float, n: int) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        elapsed = (index[:n] - index[0]).total_seconds().to_numpy() / SECONDS_PER_YEAR
        taus = tau0 - elapsed
    else:
        taus = tau0 * (1 - np.arange(n) / n)
    if np.any(taus <= 0):
        raise ValueError("underlying_path has rebalance dates at or after expiry")
    return taus


def _hedge_sigmas(hedge_delta: str, hedge_vol: pd.Series | None, iv0: float, n: int) -> np.ndarray:
    if hedge_delta == "iv_fixed":
        return np.full(n, iv0)
    if hedge_delta not in ("iv_daily", "rv"):
        raise ValueError(f"unknown hedge_delta {hedge_delta!r}")
    if hedge_vol is None:
        raise ValueError(f"hedge_delta={hedge_delta!r} needs hedge_vol")
    vols = np.asarray(hedge_vol, dtype=float)[:n]
    if len(vols) < n:
        raise ValueError("hedge_vol must cover every rebalance date")
    vols = pd.Series(np.where(vols > 0, vols, np.nan)).ffill().fillna(iv0)
    return vols.to_numpy()
