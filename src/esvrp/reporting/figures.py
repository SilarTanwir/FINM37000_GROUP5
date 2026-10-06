"""Figures for the report.

Implements PLANNING.md §6 (Planned outputs 1-7).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_figures(results: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    raise NotImplementedError
