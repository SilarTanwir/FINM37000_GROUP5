"""PLANNING.md §9: test_regimes_no_lookahead."""

import numpy as np
import pandas as pd

from esvrp.analysis.buckets import regime_labels


def test_labels_unchanged_when_later_data_appended(rng):
    idx = pd.bdate_range("2024-01-01", periods=400)
    s = pd.Series(rng.standard_normal(400).cumsum(), index=idx)
    full = regime_labels(s, window="expanding", n_bins=3)
    part = regime_labels(s.iloc[:300], window="expanding", n_bins=3)
    pd.testing.assert_series_equal(full.iloc[:300], part, check_names=False)
    assert np.isfinite(full.iloc[-1]) or full.iloc[-1] is not None


def test_rising_series_is_always_high_and_warmup_is_nan():
    s = pd.Series(np.arange(50.0), index=pd.bdate_range("2024-01-01", periods=50))
    labels = regime_labels(s, n_bins=3, min_periods=21)
    assert labels.iloc[:20].isna().all()
    assert (labels.iloc[20:] == 2).all()


def test_bins_use_history_only():
    """A low value after a high history is labeled low, whatever comes later."""
    s = pd.Series([10.0] * 30 + [1.0], index=pd.bdate_range("2024-01-01", periods=31))
    assert regime_labels(s, min_periods=21).iloc[-1] == 0
