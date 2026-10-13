"""Naive, HAC, clustered and block-bootstrap standard errors.

Implements PLANNING.md §4.6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class Estimate:
    mean: float
    se: float
    t: float
    n: int
    method: str


def default_lags(n: int) -> int:
    """Newey-West (1994) rule of thumb, floor(4 (n/100)^(2/9)), for when no horizon is given."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def mean_with_se(
    x: pd.Series | np.ndarray,
    method: Literal["naive", "hac", "cluster", "block_bootstrap"],
    lags: int | None = None,
    groups: pd.Series | np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Mean with the chosen standard error; always report naive alongside (§4.6).

    - `hac`: Newey-West (Bartlett kernel) with `lags`; §4.6 wants at least the horizon in trading days
      minus one. Without `lags`, the Newey-West rule of thumb is used.
    - `cluster`: clustered by `groups` (entry date in the hedged-gain cross-section).
    - `block_bootstrap`: moving-block bootstrap of the mean with block length `lags`
      (default n^(1/3)), `n_boot` draws and a fixed `seed` so results are reproducible.

    Missing values are dropped (together with their group).
    """
    x = np.asarray(x, dtype=float)
    keep = np.isfinite(x)
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(x):
            raise ValueError("groups must have the same length as x")
        groups = groups[keep]
    x = x[keep]
    n = len(x)
    if n < 2:
        return Estimate(float(x.mean()) if n else float("nan"), float("nan"), float("nan"), n, method)
    mean = float(x.mean())

    if method == "naive":
        se = float(x.std(ddof=1) / np.sqrt(n))
    elif method == "hac":
        fit = sm.OLS(x, np.ones(n)).fit(
            cov_type="HAC", cov_kwds={"maxlags": default_lags(n) if lags is None else lags})
        se = float(fit.bse[0])
    elif method == "cluster":
        if groups is None:
            raise ValueError("method='cluster' needs groups")
        codes = pd.factorize(groups)[0]
        fit = sm.OLS(x, np.ones(n)).fit(cov_type="cluster", cov_kwds={"groups": codes})
        se = float(fit.bse[0])
    elif method == "block_bootstrap":
        se = _block_bootstrap_se(x, lags, n_boot, seed)
    else:
        raise ValueError(f"unknown method {method!r}")
    return Estimate(mean, se, mean / se if se > 0 else float("nan"), n, method)


def _block_bootstrap_se(x: np.ndarray, block: int | None, n_boot: int, seed: int) -> float:
    n = len(x)
    block = max(1, min(n, block if block else int(np.ceil(n ** (1 / 3)))))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    return float(x[idx].mean(axis=1).std(ddof=1))
