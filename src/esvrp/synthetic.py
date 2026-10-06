"""Simulated underlying and option chains for tests and demos.

Implements PLANNING.md §9 and §5.3 (no raw Databento data is shipped).
"""

from __future__ import annotations

import pandas as pd


def simulate_underlying(n_days: int, sigma: float, seed: int = 0) -> pd.Series:
    """GBM futures path."""
    raise NotImplementedError


def simulate_chain(F: float, sigma: float, tau: float, r: float, strikes: list[float]) -> pd.DataFrame:
    """Black-76 option chain with bid/ask around the model mid."""
    raise NotImplementedError
