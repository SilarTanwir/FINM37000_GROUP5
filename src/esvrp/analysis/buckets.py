"""Moneyness, maturity and regime buckets (no look-ahead).

Implements PLANNING.md §4.5 (Moneyness and regimes).
"""

from __future__ import annotations

import pandas as pd


def regime_labels(series: pd.Series, window: str = "expanding", n_bins: int = 3) -> pd.Series:
    """Label each date by expanding-window percentile of information available at t only."""
    raise NotImplementedError
