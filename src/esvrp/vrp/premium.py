"""VRP series and constant-maturity interpolation.

Implements PLANNING.md §4.3 (maturity handling) and §4.4.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


def vrp_table(
    strips: pd.DataFrame, rv: pd.DataFrame, kind: Literal["diff", "log", "swap_return"]
) -> pd.DataFrame:
    """RV - SW, ln(RV/SW) or RV/SW - 1 (§4.4)."""
    # Decided (§15.6): RV - SW is primary; ln(RV/SW) reported alongside for inference.
    raise NotImplementedError


def constant_maturity(strips: pd.DataFrame, target_days: int = 30) -> pd.DataFrame:
    """Linear interpolation in total variance between the two expiries bracketing target_days."""
    raise NotImplementedError
