"""Monthly short delta-hedged straddle backtest and risk metrics.

Implements PLANNING.md §4.7 and §6 stage 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class StrategyResult:
    pnl: pd.Series = field(default_factory=pd.Series)
    sharpe: float = float("nan")
    skew: float = float("nan")
    max_drawdown: float = float("nan")
    cvar: float = float("nan")


def run_strategy(hedged_gains: pd.DataFrame, entry_rule: object, cost_model: object) -> StrategyResult:
    """Sell one hedged ATM straddle monthly at the bid, charge all costs, report tails."""
    # Decided (§4.7, §15.5): entry vega primary; assumed margin fraction secondary.
    raise NotImplementedError
