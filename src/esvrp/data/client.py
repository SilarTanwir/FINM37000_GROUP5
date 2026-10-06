"""Databento client wrapper with cost check and Parquet cache.

Implements PLANNING.md §5.4 (Cost and volume control). The only module allowed to read os.environ (§7.1).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class BudgetExceededError(RuntimeError):
    """Raised when an estimated pull cost exceeds data.budget_usd."""


def estimate_cost(dataset: str, schema: str, symbols: list[str], start: str, end: str) -> float:
    """Return Databento's estimated cost in USD; log it before every pull (§5.4.1)."""
    # TODO [VERIFY]: key can pull options, history available, and cost (§5.1).
    raise NotImplementedError


def fetch(
    dataset: str, schema: str, symbols: list[str], start: str, end: str,
    budget_usd: float, cache_dir: Path = Path("data"),
) -> pd.DataFrame:
    """Cost-check, then pull (or load from the Parquet cache keyed by request parameters)."""
    # TODO [VERIFY]: schema names and availability for options (§5.2).
    raise NotImplementedError
