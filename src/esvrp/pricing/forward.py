"""Parity-implied forward.

Implements PLANNING.md §4.1 (Forward price).
"""

from __future__ import annotations

import math

import pandas as pd


def parity_forward(chain_slice: pd.DataFrame, r: float, tau: float) -> float:
    """Estimate F from C - P = e^{-r tau}(F - K) at the strike where |C - P| is smallest.

    `chain_slice` is one expiry on one day with columns strike, cp and mid. Only strikes with a mid for
    both the call and the put are used; NaN if there is none.
    """
    mids = chain_slice.dropna(subset=["mid"]).groupby(["cp", "strike"])["mid"].mean()
    if not {"c", "p"} <= set(mids.index.get_level_values("cp")):
        return float("nan")
    diff = (mids["c"] - mids["p"]).dropna()
    if diff.empty:
        return float("nan")
    strike = diff.abs().idxmin()
    return float(strike + math.exp(r * tau) * diff[strike])
