"""PLANNING.md §9: test_hedge_identity (optional)."""

import numpy as np
import pandas as pd
import pytest

from esvrp.hedging.simulator import delta_hedged_gain
from esvrp.pricing.black76 import greeks, price

pytestmark = pytest.mark.xfail(reason="not implemented", strict=False)


def test_gain_matches_gamma_weighted_variance_gap(gbm_paths):
    """Pi ~ 1/2 sum Gamma_n F_n^2 (r_{n+1}^2 - sigma_i^2 dt) up to financing terms."""
    F0, K, tau, r, sig_i, n = 5000.0, 5000.0, 30 / 365, 0.04, 0.18, 21
    dt = tau / n
    paths = gbm_paths(sigma=0.25, tau=tau, n_steps=n, n_paths=200)
    opt = pd.Series({"strike": K, "cp": "c", "tau": tau, "iv": sig_i,
                     "mid": price(F0, K, tau, r, sig_i, "c")})
    actual, approx = [], []
    for p in paths:
        actual.append(delta_hedged_gain(opt, pd.Series(p), r, "iv_fixed", "mid").gain_dollars)
        total = 0.0
        for i in range(n):
            g = greeks(p[i], K, tau - i * dt, r, sig_i, "c")["gamma"]
            total += 0.5 * g * p[i] ** 2 * (np.log(p[i + 1] / p[i]) ** 2 - sig_i**2 * dt)
        approx.append(total)
    assert np.corrcoef(actual, approx)[0, 1] > 0.9
