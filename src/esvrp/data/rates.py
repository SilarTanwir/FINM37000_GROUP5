"""Risk-free rate: FRED download or constant fallback.

Implements PLANNING.md §5.5.
"""

from __future__ import annotations

import pandas as pd


def risk_free_rate(dates: pd.DatetimeIndex, source: str, constant: float) -> pd.Series:
    """Continuously compounded rate per date from FRED or the configured constant."""
    raise NotImplementedError
