"""Variance-swap strip with diagnostics.

Implements PLANNING.md §4.3 (model-free strip; truncation, discretization, jumps, stale quotes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class StripResult:
    sw: float
    k0: float
    n_strikes: int
    strike_min: float
    strike_max: float
    truncation_flag: bool
    contributions: pd.Series = field(default_factory=pd.Series)  # per-strike contributions


def variance_swap_rate(
    chain_slice: pd.DataFrame, F: float, r: float, tau: float, price_col: str = "mid"
) -> StripResult:
    """VIX-style discrete strip (baseline) with bid/mid/ask variants."""
    # See also strip.method iv_grid (Carr-Wu implementation) in §4.3.
    raise NotImplementedError
