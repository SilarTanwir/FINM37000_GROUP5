"""Underlying futures bars (5-minute and daily) for the contract that underlies each option.

Implements PLANNING.md §4.2 and §6 stage 3.
"""

from __future__ import annotations

import pandas as pd


def build_bars(raw: pd.DataFrame, freq: str, session: str) -> pd.DataFrame:
    """Resample to 5m or daily bars honoring the session definition and maintenance gap."""
    # Decided (§4.2, §15.3): full Globex; keep the return across the maintenance halt as one
    # return spanning the gap (gap rule proposed in the README PR).
    raise NotImplementedError


def settlements(stats: pd.DataFrame) -> pd.Series:
    """Daily settlement prices per underlying contract."""
    raise NotImplementedError
