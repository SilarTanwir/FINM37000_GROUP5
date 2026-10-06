"""Tables for the report.

Implements PLANNING.md §6 (Planned outputs 4 and 7).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_tables(results: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    raise NotImplementedError
