"""Delta-hedge simulator: hedge-delta variants, normalizations and time grid (roadmap 12)."""

import math

import numpy as np
import pandas as pd
import pytest

from esvrp.hedging.simulator import delta_hedged_gain, trailing_vol
from esvrp.pricing.black76 import greeks, price

F0, K, TAU, R, IV = 5000.0, 5000.0, 30 / 365, 0.04, 0.18


def _option(cp="c", **extra):
    mid = price(F0, K, TAU, R, IV, cp)
    return pd.Series({"strike": K, "cp": cp, "tau": TAU, "iv": IV, "mid": mid,
                      "bid": mid - 1.0, "ask": mid + 1.0, **extra})


def test_two_point_path_matches_formula():
    """One rebalance: Pi = payoff - e^{r tau} O_0 - Delta_0 (F_1 - F_0)."""
    opt, path = _option(), pd.Series([F0, 5100.0])
    res = delta_hedged_gain(opt, path, R, "iv_fixed", "mid")
    d0 = greeks(F0, K, TAU, R, IV, "c")["delta"]
    pi = 100.0 - math.exp(R * TAU) * opt["mid"] - d0 * 100.0
    assert res.gain_dollars == pytest.approx(pi * 50)
    assert res.gain_over_underlying == pytest.approx(pi / F0)
    assert res.gain_over_option == pytest.approx(pi / opt["mid"])
    assert res.n_hedges == 1


def test_entry_price_ask_lowers_gain_by_financed_spread():
    path = pd.Series([F0, 5050.0, 4980.0])
    mid = delta_hedged_gain(_option(), path, R, "iv_fixed", "mid").gain_dollars
    ask = delta_hedged_gain(_option(), path, R, "iv_fixed", "ask").gain_dollars
    assert mid - ask == pytest.approx(math.exp(R * TAU) * 1.0 * 50)


def test_iv_daily_at_constant_iv_equals_iv_fixed(gbm_paths):
    path = pd.Series(gbm_paths(n_paths=1)[0])
    fixed = delta_hedged_gain(_option(), path, R, "iv_fixed", "mid")
    daily = delta_hedged_gain(_option(), path, R, "iv_daily", "mid", hedge_vol=pd.Series([IV] * 22))
    assert daily.gain_dollars == pytest.approx(fixed.gain_dollars)


def test_hedge_vol_changes_the_gain_and_missing_values_carry_forward(gbm_paths):
    path = pd.Series(gbm_paths(n_paths=1)[0])
    vols = pd.Series([0.25] * 21)
    rv = delta_hedged_gain(_option(), path, R, "rv", "mid", hedge_vol=vols)
    gappy = vols.copy()
    gappy.iloc[5:9] = np.nan
    assert delta_hedged_gain(_option(), path, R, "rv", "mid", hedge_vol=gappy).gain_dollars == \
        pytest.approx(rv.gain_dollars)
    fixed = delta_hedged_gain(_option(), path, R, "iv_fixed", "mid")
    assert rv.gain_dollars != pytest.approx(fixed.gain_dollars)


def test_variants_other_than_iv_fixed_need_hedge_vol():
    with pytest.raises(ValueError):
        delta_hedged_gain(_option(), pd.Series([F0, F0]), R, "iv_daily", "mid")


def test_put_call_parity_of_hedged_gains(gbm_paths):
    """Call minus put is a forward, so their hedged gains differ by sum (1 - e^{-r tau_n}) dF_n."""
    taus = TAU * (1 - np.arange(21) / 21)
    for p in gbm_paths(n_paths=20):
        path = pd.Series(p)
        c = delta_hedged_gain(_option("c"), path, R, "iv_fixed", "mid").gain_dollars
        pp = delta_hedged_gain(_option("p"), path, R, "iv_fixed", "mid").gain_dollars
        assert c - pp == pytest.approx(50 * np.sum((1 - np.exp(-R * taus)) * np.diff(p)), abs=1e-6)


def test_dated_path_uses_calendar_time():
    """With dates, tau shrinks by elapsed ACT/365 time (over a weekend here), not by step count."""
    idx = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    path = pd.Series([F0, 5010.0, 5020.0], index=idx)
    opt = _option(tau=4 / 365)
    res = delta_hedged_gain(opt, path, R, "iv_fixed", "mid")
    d = [greeks(f, K, t / 365, R, IV, "c")["delta"] for f, t in [(F0, 4), (5010.0, 1)]]
    pi = 20.0 - math.exp(R * 4 / 365) * opt["mid"] - (d[0] * 10.0 + d[1] * 10.0)
    assert res.gain_dollars == pytest.approx(pi * 50)
    # A rebalance date at or after expiry is an error.
    with pytest.raises(ValueError):
        delta_hedged_gain(_option(tau=3 / 365), path, R, "iv_fixed", "mid")


def test_dated_hedge_vol_aligns_by_date():
    idx = pd.bdate_range("2026-01-05", periods=4)
    path = pd.Series([F0, 5030.0, 4990.0, 5010.0], index=idx)
    vols = pd.Series([0.2, 0.22, 0.24], index=idx[:3])
    shuffled = vols.iloc[[2, 0, 1]]
    a = delta_hedged_gain(_option(), path, R, "iv_daily", "mid", hedge_vol=vols)
    b = delta_hedged_gain(_option(), path, R, "iv_daily", "mid", hedge_vol=shuffled)
    assert a.gain_dollars == pytest.approx(b.gain_dollars)


def test_trailing_vol_recovers_gbm_vol(rng):
    r = 0.2 / np.sqrt(252) * rng.standard_normal(2000)
    prices = pd.Series(5000 * np.exp(np.cumsum(r)))
    vol = trailing_vol(prices, window=21)
    assert vol.iloc[:21].isna().all()
    assert vol.mean() == pytest.approx(0.2, rel=0.05)
