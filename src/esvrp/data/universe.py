"""Option universe from Databento definition records.

Implements PLANNING.md §5 and §6 stage 1. Always use the `underlying` from the definition (§5.3).
"""

from __future__ import annotations

import pandas as pd


def build_universe(definitions: pd.DataFrame) -> pd.DataFrame:
    """One row per option: strike, expiry, cp, underlying, exercise style."""
    # TODO [VERIFY]: exercise style per series (weeklies European, quarterlies American?) (§4.1).
    # TODO [DECISION]: which ES option series first: weekly | monthly_eom | quarterly (§15.1).
    raise NotImplementedError


def check_strikes_bracket_futures(universe: pd.DataFrame, futures_price: float) -> None:
    """Sanity check that strikes bracket the futures price (display-factor issue, §5.3)."""
    # TODO [VERIFY]: ES series unaffected by the strike_price display-factor issue (§5.3).
    raise NotImplementedError
