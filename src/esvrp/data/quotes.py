"""Daily quote snapshot loader and cleaning rules.

Implements PLANNING.md §6 stage 2 (cleaning rules) and §5.3.
"""

from __future__ import annotations

import pandas as pd


def load_snapshots(universe: pd.DataFrame, bbo: pd.DataFrame, snapshot_time_ct: str) -> pd.DataFrame:
    """Last bid/ask per option before the snapshot time, one row per option-day."""
    # TODO [DECISION]: snapshot time and ES settlement-time convention (§15.4).
    raise NotImplementedError


def clean_quotes(quotes: pd.DataFrame, cfg: object) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply cleaning rules; return cleaned quotes and a per-rule drop count log."""
    raise NotImplementedError
