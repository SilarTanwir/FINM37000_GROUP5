"""Underlying futures bars (5-minute and daily) for the contract that underlies each option.

Implements PLANNING.md §4.2 and §6 stage 3.
"""

from __future__ import annotations

import pandas as pd


def build_bars(raw: pd.DataFrame, freq: str, session: str) -> pd.DataFrame:
    """Resample to 5m or daily bars honoring the session definition and maintenance gap."""
    # TODO [DECISION]: full Globex vs regular hours, and the maintenance-gap treatment (§4.2, §15.3).
    raise NotImplementedError


def settlements(stats: pd.DataFrame) -> pd.Series:
    """Daily settlement prices per underlying contract."""
    # TODO [VERIFY]: settlement stat type exists for options/futures in `statistics` (§5.2).
    raise NotImplementedError
