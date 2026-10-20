"""PLANNING.md §9: test_costs."""

import pytest

from esvrp.hedging.costs import apply_costs


def test_costs_reduce_gain():
    gross = 100.0
    net = apply_costs(gross, option_halfspread=0.5, n_hedges=21, slippage_ticks=0.5,
                      fee_per_contract=1.50)
    assert net < gross


def test_costs_scale_with_hedges_and_fees():
    base = apply_costs(0.0, 0.5, 21, 0.5, 1.50)
    more = apply_costs(0.0, 0.5, 42, 0.5, 1.50)
    assert more < base
    assert apply_costs(0.0, 0.0, 0, 0.0, 0.0) == pytest.approx(0.0)


def test_tick_value():
    """One tick = 0.25 points x $50 = $12.50; 0.5 tick slippage on one hedge costs $6.25."""
    assert apply_costs(0.0, 0.0, 1, 0.5, 0.0) == pytest.approx(-6.25)


def test_cost_breakdown():
    """1 point half-spread = $50; 21 hedges x 0.5 tick = $131.25; 22 trades x $1.50 fees = $33."""
    assert apply_costs(0.0, 1.0, 21, 0.5, 1.50) == pytest.approx(-(50.0 + 131.25 + 33.0))
