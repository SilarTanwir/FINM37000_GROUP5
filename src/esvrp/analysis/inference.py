"""Naive, HAC, clustered and block-bootstrap standard errors.

Implements PLANNING.md §4.6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class Estimate:
    mean: float
    se: float
    t: float
    n: int
    method: str


def mean_with_se(
    x: pd.Series | np.ndarray,
    method: Literal["naive", "hac", "cluster", "block_bootstrap"],
    lags: int | None = None,
    groups: pd.Series | np.ndarray | None = None,
) -> Estimate:
    """Mean with the chosen standard error; always report naive alongside (§4.6)."""
    raise NotImplementedError
