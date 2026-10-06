"""Shared offline simulation fixtures. No test may call the Databento API or need a key."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def gbm_paths(rng):
    """Factory: (n_paths, n_steps+1) GBM futures paths with zero drift (futures are Q-martingales)."""

    def make(F0=5000.0, sigma=0.18, tau=30 / 365, n_steps=21, n_paths=500):
        dt = tau / n_steps
        z = rng.standard_normal((n_paths, n_steps))
        log_ret = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z
        return F0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_ret, axis=1)], 1))

    return make


@pytest.fixture
def flat_vol_chain():
    """Factory: dense OTM strike grid priced at constant sigma (Black-76)."""
    from esvrp.pricing.black76 import price

    def make(F=5000.0, sigma=0.2, tau=30 / 365, r=0.04, lo=0.5, hi=1.6, step=5.0):
        strikes = np.arange(F * lo, F * hi, step)
        rows = []
        for k in strikes:
            cp = "p" if k < F else "c"
            p = price(F, float(k), tau, r, sigma, cp)
            rows.append({"strike": float(k), "cp": cp, "mid": p, "bid": p, "ask": p})
        return pd.DataFrame(rows), F, r, tau

    return make
