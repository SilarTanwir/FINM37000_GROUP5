"""PLANNING.md §9: test_inference."""

import numpy as np
import pytest

from esvrp.analysis.inference import mean_with_se


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


def test_naive_matches_formula(rng):
    x = rng.standard_normal(500)
    est = mean_with_se(x, "naive")
    assert est.se == pytest.approx(x.std(ddof=1) / np.sqrt(500))
    assert est.t == pytest.approx(est.mean / est.se)
    assert est.n == 500


def test_hac_with_zero_lags_matches_naive_up_to_dof(rng):
    x = rng.standard_normal(500)
    hac = mean_with_se(x, "hac", lags=0)
    naive = mean_with_se(x, "naive")
    assert hac.se == pytest.approx(naive.se, rel=0.01)


def test_block_bootstrap_exceeds_naive_and_is_reproducible(rng):
    x = _ar1(rng)
    a = mean_with_se(x, "block_bootstrap", lags=30)
    assert a.se > mean_with_se(x, "naive").se
    assert a.se == mean_with_se(x, "block_bootstrap", lags=30).se


def test_missing_values_dropped(rng):
    x = rng.standard_normal(100)
    x[[3, 50]] = np.nan
    assert mean_with_se(x, "naive").n == 98


def test_cluster_needs_groups(rng):
    with pytest.raises(ValueError):
        mean_with_se(rng.standard_normal(10), "cluster")
