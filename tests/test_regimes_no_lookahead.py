"""PLANNING.md §9: test_regimes_no_lookahead."""

import numpy as np
import pandas as pd
import pytest

from esvrp.analysis.buckets import regime_labels

pytestmark = pytest.mark.xfail(reason="not implemented", strict=False)


def test_labels_unchanged_when_later_data_appended(rng):
    idx = pd.bdate_range("2024-01-01", periods=400)
    s = pd.Series(rng.standard_normal(400).cumsum(), index=idx)
    full = regime_labels(s, window="expanding", n_bins=3)
    part = regime_labels(s.iloc[:300], window="expanding", n_bins=3)
    pd.testing.assert_series_equal(full.iloc[:300], part, check_names=False)
    assert np.isfinite(full.iloc[-1]) or full.iloc[-1] is not None
