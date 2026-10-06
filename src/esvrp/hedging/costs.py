"""Spread, slippage and fee model.

Implements PLANNING.md §4.7 (Costs). ES tick = 0.25 index points = $12.50 ($50 multiplier).
"""

from __future__ import annotations


def apply_costs(
    gross: float, option_halfspread: float, n_hedges: int, slippage_ticks: float,
    fee_per_contract: float,
) -> float:
    """Net gain after buying at ask / selling at bid, hedge slippage and per-contract fees."""
    raise NotImplementedError
