"""Black-76 pricing, Greeks and implied volatility for options on futures.

Implements PLANNING.md §4.1 (Options on futures: Black-76). Greeks are with respect to F.
``cp`` is "c" for calls and "p" for puts. ``tau`` is in years (ACT/365).

`price` and `greeks` take scalars or arrays (broadcast together) and return floats for scalar inputs,
so a whole chain is priced in one call. `implied_vol` solves one option and `implied_vols` applies it
to arrays. Inputs outside the model's domain (F or K not positive, missing values) give NaN instead of
an exception, so one bad quote cannot stop a chain.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

_SIGMA_LO = 1e-6
_SIGMA_HI = 10.0


def _is_call(cp: Any) -> np.ndarray:
    cp = np.char.lower(np.asarray(cp, dtype=str))
    if not np.isin(cp, ("c", "p")).all():
        raise ValueError(f"cp must be 'c' or 'p', got {np.unique(cp).tolist()}")
    return cp == "c"


def _d1_sd(F: np.ndarray, K: np.ndarray, tau: np.ndarray, sigma: np.ndarray) -> tuple[Any, Any, Any]:
    """d1, sigma*sqrt(tau), and which options are live (tau > 0 and sigma > 0).

    Options that are not live get a placeholder sd of 1; callers replace their values with the limit.
    """
    live = (tau > 0) & (sigma > 0)
    sd = np.where(live, sigma * np.sqrt(np.where(live, tau, 1.0)), 1.0)
    return (np.log(F / K) + 0.5 * sd**2) / sd, sd, live


def _cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _finish(x: np.ndarray, valid: np.ndarray) -> Any:
    x = np.where(valid, x, np.nan)
    return float(x) if x.ndim == 0 else x


def price(F: Any, K: Any, tau: Any, r: Any, sigma: Any, cp: Any) -> Any:
    """Black-76 price of a European option on a futures contract (upfront premium).

    At expiry (tau <= 0) the price is intrinsic value; with no volatility it is discounted intrinsic.
    """
    call = _is_call(cp)
    F, K, tau, r, sigma = (np.asarray(x, dtype=float) for x in (F, K, tau, r, sigma))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, sd, live = _d1_sd(F, K, tau, sigma)
        d2 = d1 - sd
        model = np.where(call, F * ndtr(d1) - K * ndtr(d2), K * ndtr(-d2) - F * ndtr(-d1))
        intrinsic = np.maximum(np.where(call, F - K, K - F), 0.0)
        value = np.exp(-r * np.maximum(tau, 0.0)) * np.where(live, model, intrinsic)
    return _finish(value, (F > 0) & (K > 0) & ~np.isnan(tau + sigma))


def greeks(F: Any, K: Any, tau: Any, r: Any, sigma: Any, cp: Any) -> dict[str, Any]:
    """Delta, gamma and vega with respect to the futures price F and sigma.

    At expiry or with no volatility, delta is the exercise indicator and gamma and vega are zero.
    """
    call = _is_call(cp)
    F, K, tau, r, sigma = (np.asarray(x, dtype=float) for x in (F, K, tau, r, sigma))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, sd, live = _d1_sd(F, K, tau, sigma)
        disc = np.exp(-r * np.maximum(tau, 0.0))
        pdf = np.where(live, np.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi), 0.0)
        delta_c = disc * np.where(live, ndtr(d1), F > K)
        delta = np.where(call, delta_c, delta_c - disc)
        gamma = disc * pdf / (F * sd)
        vega = F * disc * pdf * np.sqrt(np.maximum(tau, 0.0))
    valid = (F > 0) & (K > 0) & ~np.isnan(tau + sigma)
    return {"delta": _finish(delta, valid), "gamma": _finish(gamma, valid), "vega": _finish(vega, valid)}


def implied_vol(
    price_: float, F: float, K: float, tau: float, r: float, cp: str, tol: float = 1e-8
) -> float:
    """Implied volatility by Brent root-finding; NaN if the price violates no-arbitrage bounds.

    Rejects prices at or below intrinsic value and above the upper bound
    (``e^{-r tau} F`` for calls, ``e^{-r tau} K`` for puts). Also NaN for missing inputs, at expiry,
    and when the volatility lies outside the solver's bracket (1e-6 to 10).
    """
    call = bool(_is_call(cp))
    try:
        price_, F, K, tau, r = (float(x) for x in (price_, F, K, tau, r))
    except (TypeError, ValueError):
        return float("nan")
    if not all(math.isfinite(x) for x in (price_, F, K, tau, r)) or F <= 0 or K <= 0 or tau <= 0:
        return float("nan")
    disc = math.exp(-r * tau)
    if call:
        intrinsic, upper = disc * max(F - K, 0.0), disc * F
    else:
        intrinsic, upper = disc * max(K - F, 0.0), disc * K
    if price_ <= intrinsic + 1e-12 or price_ >= upper:
        return float("nan")

    log_fk, sqrt_tau, sign = math.log(F / K), math.sqrt(tau), 1.0 if call else -1.0

    def objective(s: float) -> float:  # scalar form of `price`, much faster inside the solver
        sd = s * sqrt_tau
        d1 = log_fk / sd + 0.5 * sd
        return disc * sign * (F * _cdf(sign * d1) - K * _cdf(sign * (d1 - sd))) - price_

    if objective(_SIGMA_LO) > 0 or objective(_SIGMA_HI) < 0:
        return float("nan")
    root, res = brentq(objective, _SIGMA_LO, _SIGMA_HI, xtol=tol, rtol=1e-12, full_output=True, disp=False)
    return float(root) if res.converged else float("nan")


def implied_vols(price_: Any, F: Any, K: Any, tau: Any, r: Any, cp: Any, tol: float = 1e-8) -> np.ndarray:
    """`implied_vol` for arrays of options (broadcast together): one Brent solve per option."""
    num = [np.asarray(x, dtype=float) for x in (price_, F, K, tau, r)]
    *num, cp = np.broadcast_arrays(*num, np.asarray(cp, dtype=str))
    rows = zip(*(a.ravel() for a in num), cp.ravel(), strict=True)
    return np.array([implied_vol(*row, tol=tol) for row in rows]).reshape(cp.shape)
