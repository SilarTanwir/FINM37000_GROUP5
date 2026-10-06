"""Delta-hedged gain engine.

Implements PLANNING.md §4.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass
class HedgeResult:
    gain_dollars: float
    gain_over_underlying: float
    gain_over_option: float
    n_hedges: int


def delta_hedged_gain(
    option: pd.Series,
    underlying_path: pd.Series,
    rate: float,
    hedge_delta: Literal["iv_daily", "iv_fixed", "rv"],
    entry_price: Literal["mid", "ask", "bid"],
) -> HedgeResult:
    """Pi = O_T - e^{r tau} O_0 - sum_n Delta_n (F_{n+1} - F_n)."""
    # TODO [VERIFY]: identity and financing terms in the derivation doc (§4.5).
    raise NotImplementedError
