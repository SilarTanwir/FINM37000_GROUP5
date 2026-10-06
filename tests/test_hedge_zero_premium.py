"""PLANNING.md §9: test_hedge_zero_premium."""

import numpy as np
import pandas as pd
import pytest

from esvrp.hedging.simulator import delta_hedged_gain
from esvrp.pricing.black76 import price

pytestmark = pytest.mark.xfail(reason="not implemented", strict=False)

F0, K, TAU, R = 5000.0, 5000.0, 30 / 365, 0.04


def _mean_gain(paths, sigma_price):
    opt = pd.Series({"strike": K, "cp": "c", "tau": TAU, "iv": sigma_price,
                     "mid": price(F0, K, TAU, R, sigma_price, "c")})
    gains = [delta_hedged_gain(opt, pd.Series(p), R, "iv_fixed", "mid").gain_dollars for p in paths]
    return np.mean(gains), np.std(gains) / np.sqrt(len(gains))


def test_zero_when_realized_equals_implied(gbm_paths):
    m, se = _mean_gain(gbm_paths(sigma=0.18), 0.18)
    assert abs(m) < 3 * se


def test_long_option_gains_when_realized_exceeds_implied(gbm_paths):
    m, se = _mean_gain(gbm_paths(sigma=0.30), 0.18)
    assert m > 3 * se
