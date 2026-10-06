"""PLANNING.md §9: test_realized_variance."""

import numpy as np
import pandas as pd
import pytest

from esvrp.vrp.realized import realized_variance

pytestmark = pytest.mark.xfail(reason="not implemented", strict=False)


@pytest.mark.parametrize("n_steps", [21, 21 * 78])  # daily and 5-minute-like sampling
def test_rv_converges_to_sigma_squared(gbm_paths, n_steps):
    sigma, tau = 0.18, 30 / 365
    paths = gbm_paths(sigma=sigma, tau=tau, n_steps=n_steps, n_paths=200)
    rvs = [realized_variance(pd.Series(np.diff(np.log(p))), tau) for p in paths]
    assert np.mean(rvs) == pytest.approx(sigma**2, rel=0.03)


def test_gap_handling(gbm_paths):
    """The overnight/maintenance gap must not be double counted (config session)."""
    raise NotImplementedError("fixture for session gaps not written yet")
