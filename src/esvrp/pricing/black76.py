"""Black-76 pricing, Greeks and implied volatility for options on futures.

Implements PLANNING.md §4.1 (Options on futures: Black-76). Greeks are with respect to F.
``cp`` is "c" for calls and "p" for puts. ``tau`` is in years (ACT/365).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

_SIGMA_LO = 1e-6
_SIGMA_HI = 10.0


def _check_cp(cp: str) -> str:
    cp = cp.lower()
    if cp not in ("c", "p"):
        raise ValueError(f"cp must be 'c' or 'p', got {cp!r}")
    return cp


def _d1_d2(F: float, K: float, tau: float, sigma: float) -> tuple[float, float]:
    sd = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * tau) / sd
    return d1, d1 - sd


def price(F: float, K: float, tau: float, r: float, sigma: float, cp: str) -> float:
    """Black-76 price of a European option on a futures contract (upfront premium)."""
    cp = _check_cp(cp)
    disc = math.exp(-r * tau)
    if tau <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if cp == "c" else max(K - F, 0.0)
        return disc * intrinsic if tau > 0 else intrinsic
    d1, d2 = _d1_d2(F, K, tau, sigma)
    if cp == "c":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def greeks(F: float, K: float, tau: float, r: float, sigma: float, cp: str) -> dict[str, float]:
    """Delta, gamma and vega with respect to the futures price F and sigma."""
    cp = _check_cp(cp)
    disc = math.exp(-r * tau)
    d1, _ = _d1_d2(F, K, tau, sigma)
    sqrt_tau = math.sqrt(tau)
    delta_c = disc * norm.cdf(d1)
    delta = delta_c if cp == "c" else delta_c - disc
    gamma = disc * norm.pdf(d1) / (F * sigma * sqrt_tau)
    vega = F * disc * norm.pdf(d1) * sqrt_tau
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}


def implied_vol(
    price_: float, F: float, K: float, tau: float, r: float, cp: str, tol: float = 1e-8
) -> float:
    """Implied volatility by Brent root-finding; NaN if the price violates no-arbitrage bounds.

    Rejects prices at or below intrinsic value and above the upper bound
    (``e^{-r tau} F`` for calls, ``e^{-r tau} K`` for puts).
    """
    cp = _check_cp(cp)
    if not all(np.isfinite([price_, F, K, tau, r])) or F <= 0 or K <= 0 or tau <= 0:
        return float("nan")
    disc = math.exp(-r * tau)
    if cp == "c":
        intrinsic, upper = disc * max(F - K, 0.0), disc * F
    else:
        intrinsic, upper = disc * max(K - F, 0.0), disc * K
    if price_ <= intrinsic + 1e-12 or price_ >= upper:
        return float("nan")

    def objective(s: float) -> float:
        return price(F, K, tau, r, s, cp) - price_

    if objective(_SIGMA_LO) > 0 or objective(_SIGMA_HI) < 0:
        return float("nan")
    return float(brentq(objective, _SIGMA_LO, _SIGMA_HI, xtol=tol, rtol=1e-12))
