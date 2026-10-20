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


@pytest.fixture
def raw_week():
    """Factory: 1-minute records of one futures contract in full-Globex sessions (stage 3 raw format).

    Sessions run 17:00 CT to 16:00 CT the next day from Thu 2026-10-08 evening through Fri 2026-10-16
    (CDT), with a halt each day and no trading from Fri 16:00 CT to Sun 17:00 CT.
    """

    def make(seed: int = 7, price: float = 5000.0, per_minute_vol: float = 0.0004, symbol: str = "ESZ6"):
        rng = np.random.default_rng(seed)
        tz = "America/Chicago"
        sessions = []
        for open_day in ["2026-10-08", "2026-10-11", "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]:
            start = pd.Timestamp(f"{open_day} 17:00", tz=tz)
            end = start + pd.Timedelta(hours=23)
            sessions.append(pd.date_range(start, end, freq="1min", inclusive="left"))
        ts = sessions[0].append(sessions[1:]).tz_convert("UTC")
        close = price * np.exp(np.cumsum(rng.standard_normal(len(ts)) * per_minute_vol))
        return pd.DataFrame({"symbol": symbol, "ts_event": ts, "open": close, "high": close, "low": close,
                             "close": close, "volume": 1})

    return make
