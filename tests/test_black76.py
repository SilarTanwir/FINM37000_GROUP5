"""Tests for pricing/black76.py (PLANNING.md §9, test_black76)."""

import math

import numpy as np
import pytest

from esvrp.pricing.black76 import greeks, implied_vol, implied_vols, price

F, K, TAU, R, SIGMA = 5000.0, 5100.0, 30 / 365, 0.04, 0.18


def test_put_call_parity():
    c = price(F, K, TAU, R, SIGMA, "c")
    p = price(F, K, TAU, R, SIGMA, "p")
    assert c - p == pytest.approx(math.exp(-R * TAU) * (F - K), abs=1e-9)


@pytest.mark.parametrize("cp", ["c", "p"])
@pytest.mark.parametrize("sigma", [0.05, 0.18, 0.6, 1.5])
@pytest.mark.parametrize("strike", [4500.0, 5000.0, 5500.0])
def test_iv_round_trip(cp, sigma, strike):
    p = price(F, strike, TAU, R, sigma, cp)
    intrinsic = math.exp(-R * TAU) * (max(F - strike, 0.0) if cp == "c" else max(strike - F, 0.0))
    if p - intrinsic < 0.5:
        pytest.skip("time value too small for a well-conditioned inversion")
    assert implied_vol(p, F, strike, TAU, R, cp) == pytest.approx(sigma, rel=1e-5)


@pytest.mark.parametrize("cp", ["c", "p"])
def test_greeks_match_finite_differences(cp):
    g = greeks(F, K, TAU, R, SIGMA, cp)
    h = 0.01
    up, mid, dn = (price(F + d, K, TAU, R, SIGMA, cp) for d in (h, 0.0, -h))
    assert g["delta"] == pytest.approx((up - dn) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((up - 2 * mid + dn) / h**2, rel=1e-4)
    hs = 1e-5
    vega_fd = (price(F, K, TAU, R, SIGMA + hs, cp) - price(F, K, TAU, R, SIGMA - hs, cp)) / (2 * hs)
    assert g["vega"] == pytest.approx(vega_fd, rel=1e-6)


def test_delta_put_call_relation():
    dc = greeks(F, K, TAU, R, SIGMA, "c")["delta"]
    dp = greeks(F, K, TAU, R, SIGMA, "p")["delta"]
    assert dc - dp == pytest.approx(math.exp(-R * TAU), abs=1e-12)


def test_iv_rejects_arbitrage_violations():
    disc = math.exp(-R * TAU)
    assert math.isnan(implied_vol(0.0, F, K, TAU, R, "c"))  # zero price
    assert math.isnan(implied_vol(disc * (F - 4900.0) - 1.0, F, 4900.0, TAU, R, "c"))  # < intrinsic
    assert math.isnan(implied_vol(disc * F + 1.0, F, K, TAU, R, "c"))  # > upper bound
    assert math.isnan(implied_vol(disc * K + 1.0, F, K, TAU, R, "p"))
    assert math.isnan(implied_vol(float("nan"), F, K, TAU, R, "c"))
    assert math.isnan(implied_vol(10.0, F, K, 0.0, R, "c"))


def test_invalid_cp():
    with pytest.raises(ValueError):
        price(F, K, TAU, R, SIGMA, "x")


def test_price_increases_with_vol():
    vols = np.linspace(0.05, 0.8, 8)
    prices = [price(F, K, TAU, R, s, "c") for s in vols]
    assert all(b > a for a, b in zip(prices, prices[1:], strict=False))


def test_arrays_match_scalars():
    strikes, cps = np.array([4500.0, 5000.0, 5500.0]), np.array(["p", "c", "c"])
    p = price(F, strikes, TAU, R, SIGMA, cps)
    g = greeks(F, strikes, TAU, R, SIGMA, cps)
    for i, (k, cp) in enumerate(zip(strikes, cps, strict=True)):
        assert p[i] == pytest.approx(price(F, k, TAU, R, SIGMA, cp), rel=1e-12)
        assert {name: v[i] for name, v in g.items()} == pytest.approx(greeks(F, k, TAU, R, SIGMA, cp))
    assert isinstance(price(F, K, TAU, R, SIGMA, "c"), float)      # scalars in, float out
    assert implied_vols(p, F, strikes, TAU, R, cps) == pytest.approx([SIGMA] * 3, rel=1e-5)


def test_expiry_and_zero_vol_limits():
    disc = math.exp(-R * TAU)
    assert price(F, K, 0.0, R, SIGMA, "p") == K - F                 # at expiry: intrinsic, not discounted
    assert price(F, K, 0.0, R, SIGMA, "c") == 0.0
    assert price(F, K, TAU, R, 0.0, "p") == pytest.approx(disc * (K - F), rel=1e-12)
    assert greeks(F, K, 0.0, R, SIGMA, "p") == {"delta": -1.0, "gamma": 0.0, "vega": 0.0}
    assert greeks(F, K, 0.0, R, SIGMA, "c") == {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
    assert greeks(F, K, TAU, R, 0.0, "p")["delta"] == pytest.approx(-disc, rel=1e-12)


def test_bad_inputs_give_nan_not_errors():
    nan = float("nan")
    assert math.isnan(price(0.0, K, TAU, R, SIGMA, "c"))            # log(F/K) undefined
    assert math.isnan(price(F, -5100.0, TAU, R, SIGMA, "p"))
    assert math.isnan(price(F, K, TAU, R, nan, "c"))
    assert all(math.isnan(v) for v in greeks(nan, K, TAU, R, SIGMA, "c").values())
    assert math.isnan(implied_vol(None, F, K, TAU, R, "c"))
    assert math.isnan(implied_vol(10.0, 0.0, K, TAU, R, "c"))
    with pytest.raises(ValueError):
        implied_vol(10.0, F, K, TAU, R, "x")


def test_iv_edge_cases():
    for strike, cp in [(2000.0, "p"), (9000.0, "c")]:                # one-tick quotes in the far wings
        iv = implied_vol(0.05, F, strike, TAU, R, cp)
        assert price(F, strike, TAU, R, iv, cp) == pytest.approx(0.05, rel=1e-6)
    minute = 60 / (365 * 86400)                                     # one minute before expiry
    assert implied_vol(price(F, 5005.0, minute, R, 0.2, "c"), F, 5005.0, minute, R, "c") == pytest.approx(0.2)
    assert implied_vol(price(F, 5001.0, TAU, R, 0.001, "c"), F, 5001.0, TAU, R, "c") == pytest.approx(0.001)
    assert math.isnan(implied_vol(price(F, K, TAU, R, 12.0, "c"), F, K, TAU, R, "c"))  # above the bracket


def test_implied_vols_keeps_going_past_bad_quotes():
    good = price(F, K, TAU, R, SIGMA, "c")
    iv = implied_vols([good, 0.0, float("nan"), good], F, K, [TAU, TAU, TAU, 0.0], R, ["c", "c", "p", "c"])
    assert iv[0] == pytest.approx(SIGMA, rel=1e-5)
    assert np.isnan(iv[1:]).all()                                   # zero price, missing price, expired
