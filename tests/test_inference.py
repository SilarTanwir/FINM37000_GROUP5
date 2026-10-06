"""PLANNING.md §9: test_inference."""

import numpy as np
import pytest

from esvrp.analysis.inference import mean_with_se

pytestmark = pytest.mark.xfail(reason="not implemented", strict=False)


def _ar1(rng, n=1500, phi=0.95):
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_hac_se_exceeds_naive_on_overlapping_series(rng):
    x = _ar1(rng)
    naive = mean_with_se(x, "naive")
    hac = mean_with_se(x, "hac", lags=30)
    assert hac.se > naive.se


def test_cluster_se_exceeds_naive_with_common_shocks(rng):
    day_shock = np.repeat(rng.standard_normal(100), 20)
    x = day_shock + 0.1 * rng.standard_normal(2000)
    groups = np.repeat(np.arange(100), 20)
    assert mean_with_se(x, "cluster", groups=groups).se > mean_with_se(x, "naive").se
