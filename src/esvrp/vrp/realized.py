"""Realized variance (daily and 5-minute).

Implements PLANNING.md §4.2.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


def realized_variance(returns: pd.Series, tau: float) -> float:
    """RV = (1/tau) * sum(r_i^2), annualized on the same basis as implied variance."""
    raise NotImplementedError


def returns_from_bars(bars: pd.DataFrame, freq: Literal["5m", "1d"], session: str) -> pd.Series:
    """Log returns of the specific underlying contract."""
    # TODO [DECISION]: session definition and maintenance gap (§4.2, §15.3).
    raise NotImplementedError
