"""Moneyness, maturity and regime buckets (no look-ahead).

Implements PLANNING.md §4.5 (Moneyness and regimes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_labels(
    series: pd.Series, window: str = "expanding", n_bins: int = 3, min_periods: int = 21,
) -> pd.Series:
    """Label each date by expanding-window percentile of information available at t only.

    The percentile of x_t is the share of observations up to and including t that are <= x_t. Labels
    run from 0 (lowest bin) to n_bins - 1, so with 3 bins a date is "high" (2) if x_t is above two
    thirds of the history so far. Dates with a missing value or fewer than `min_periods` observations
    of history get NaN. Appending later data never changes an earlier label.
    """
    if window != "expanding":
        raise ValueError(f"only window='expanding' is supported, got {window!r}")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    values = series.to_numpy(dtype=float)
    labels = np.full(len(values), np.nan)
    seen: list[float] = []
    for i, v in enumerate(values):
        if not np.isfinite(v):
            continue
        seen.append(v)
        if len(seen) < min_periods:
            continue
        pct = np.mean(np.asarray(seen) <= v)
        labels[i] = np.ceil(pct * n_bins) - 1
    return pd.Series(labels, index=series.index, name=series.name)
