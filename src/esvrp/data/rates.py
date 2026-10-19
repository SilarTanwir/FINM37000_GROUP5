"""Risk-free rate: FRED download or constant fallback.

Implements PLANNING.md §5.5.
"""

from __future__ import annotations

import pandas as pd


def risk_free_rate(dates: pd.DatetimeIndex, source: str, constant: float) -> pd.Series:
    """Continuously compounded rate per date from FRED or the configured constant."""
    if source == "constant":
        return pd.Series(float(constant), index=dates, name="r")
    # The FRED download (§5.5) is optional and not written yet; the constant is the planned fallback.
    raise NotImplementedError(f"rate source {source!r} is not implemented yet; use rate_source: constant")
