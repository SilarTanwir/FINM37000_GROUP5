"""Tests for pricing/black76.py (PLANNING.md §9, test_black76)."""

import math

import numpy as np
import pytest

from esvrp.pricing.black76 import greeks, implied_vol, price

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
