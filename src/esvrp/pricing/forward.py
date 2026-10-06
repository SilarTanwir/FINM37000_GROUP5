"""Parity-implied forward.

Implements PLANNING.md §4.1 (Forward price).
"""

from __future__ import annotations

import pandas as pd


def parity_forward(chain_slice: pd.DataFrame, r: float, tau: float) -> float:
    """Estimate F from C - P = e^{-r tau}(F - K) at the strike where |C - P| is smallest."""
    raise NotImplementedError
