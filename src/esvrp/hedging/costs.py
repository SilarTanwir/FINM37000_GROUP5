"""Spread, slippage and fee model.

Implements PLANNING.md §4.7 (Costs). ES tick = 0.25 index points = $12.50 ($50 multiplier).

All amounts are in dollars per option contract. The option is held to expiry, so it crosses the spread
once (at entry, half-spread versus mid) and settles at intrinsic value with no exit spread. Each hedge
rebalance trades one futures contract (one ES option is on one ES future), paying slippage and a fee.
"""

from __future__ import annotations

ES_MULTIPLIER = 50.0  # dollars per index point
ES_TICK = 0.25  # index points


def apply_costs(
    gross: float, option_halfspread: float, n_hedges: int, slippage_ticks: float,
    fee_per_contract: float,
) -> float:
    """Net gain after buying at ask / selling at bid, hedge slippage and per-contract fees.

    `gross` is the mid-based gain in dollars, `option_halfspread` is in index points and
    `fee_per_contract` in dollars. The sign of the trade does not matter: costs always reduce the gain.
    """
    spread = option_halfspread * ES_MULTIPLIER
    slippage = n_hedges * slippage_ticks * ES_TICK * ES_MULTIPLIER
    fees = (1 + n_hedges) * fee_per_contract  # the option trade plus each hedge trade
    return gross - spread - slippage - fees
