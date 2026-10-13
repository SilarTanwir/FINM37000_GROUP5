"""Stage 5 entry point: variance swap strip, realized variance and the variance risk premium.

Implements PLANNING.md §6 stage 5 (§4.2-4.4). Reads the chains of stage 4 and the bars of stage 3, and
writes, under `data_dir`:

- `strip.parquet`: one row per chain, `sw_mid` / `sw_bid` / `sw_ask` and strip diagnostics (roadmap 9)
- `rv.parquet`: one row per chain, 5-minute and daily realized variance, log and simple (roadmap 10)
- `vrp.parquet`: the premium per chain, RV - SW primary (roadmap 11), at each expiry's own maturity
- `strip_cm.parquet`, `rv_cm.parquet`, `vrp_cm.parquet`: the constant 30-day series (Carr-Wu comparable)

and under `report_dir`: `tables/vrp_descriptives.csv`, `tables/vrp_cm_descriptives.csv` and the figures
`figures/sw_vs_rv.png`, `vrp_distribution.png`, `rv_comparison.png` (every chain) and
`figures/cm_sw_vs_rv.png`, `cm_vrp_distribution.png` (constant maturity). No data is pulled.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from esvrp.data.client import DATA_DIR, REPO_ROOT
from esvrp.reporting.figures import (
    rv_comparison_figure,
    sw_vs_rv_figure,
    vrp_distribution_figure,
)
from esvrp.vrp.premium import (
    constant_maturity,
    constant_maturity_rv,
    vrp_descriptives,
    vrp_table,
)
from esvrp.vrp.realized import rv_table
from esvrp.vrp.strip import strip_table

log = logging.getLogger(__name__)

REPORT_DIR = REPO_ROOT / "reports"
TARGET_DAYS = 30
CHAIN_KEYS = ["date", "expiry", "underlying", "snapshot_ts", "tau"]


def run(cfg: Any, data_dir: Path = DATA_DIR, report_dir: Path = REPORT_DIR) -> dict[str, pd.DataFrame]:
    """Compute and save the stage 5 tables and figures; return the tables by name."""
    if cfg.strip.method != "vix_discrete":
        raise NotImplementedError(f"strip.method {cfg.strip.method!r} is not implemented (see issue #31)")
    data_dir, report_dir = Path(data_dir), Path(report_dir)
    chain = pd.read_parquet(data_dir / "chain.parquet")
    if chain.empty:
        raise ValueError("chain.parquet has no rows; run stages 1-4 first")
    bars_5m = pd.read_parquet(data_dir / "underlying_5m.parquet")
    daily = pd.read_parquet(data_dir / "underlying_daily.parquet")

    sw_col = f"sw_{cfg.strip.price_col}"
    rv_col = "rv_5m" if "5m" in cfg.realized.frequency else "rv_daily"

    # Each expiry at its own maturity (matched horizon, §4.3).
    strips = strip_table(chain, cfg.cleaning.min_range_pct, cfg.strip.min_iv, cfg.strip.max_iv)
    rv = rv_table(chain[CHAIN_KEYS], bars_5m, daily, cfg.realized.max_gap_days)
    vrp = vrp_table(strips, rv, "diff", sw_col=sw_col, rv_col=rv_col)

    # Constant 30-day maturity, the series comparable to Carr-Wu (Eq. 10).
    cm = constant_maturity(strips, TARGET_DAYS, cfg.strip.min_front_days)
    rv_cm = constant_maturity_rv(cm, bars_5m, daily, cfg.realized.max_gap_days)
    vrp_cm = vrp_table(cm, rv_cm, "diff", sw_col=sw_col, rv_col=rv_col) if len(cm) else cm

    tables = {"strip": strips, "rv": rv, "vrp": vrp, "strip_cm": cm, "rv_cm": rv_cm, "vrp_cm": vrp_cm}
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(data_dir / f"{name}.parquet")

    (report_dir / "tables").mkdir(parents=True, exist_ok=True)
    vrp_descriptives(vrp).to_csv(report_dir / "tables" / "vrp_descriptives.csv")
    figures = report_dir / "figures"
    sw_vs_rv_figure(vrp, figures / "sw_vs_rv.png")
    vrp_distribution_figure(vrp, figures / "vrp_distribution.png")
    rv_comparison_figure(rv, figures / "rv_comparison.png")
    if len(vrp_cm):
        vrp_descriptives(vrp_cm).to_csv(report_dir / "tables" / "vrp_cm_descriptives.csv")
        sw_vs_rv_figure(vrp_cm, figures / "cm_sw_vs_rv.png")
        vrp_distribution_figure(vrp_cm, figures / "cm_vrp_distribution.png")

    log.info("Stage 5: %d chains, %d with a premium (mean RV - SW %.5f); %d chains left out because the bars "
             "do not cover the window (max_gap_days=%s); %d quotes dropped by the implied-vol bounds; "
             "%d constant-maturity dates (%d extrapolated, %d with a premium) -> %s",
             len(strips), int(vrp["vrp_diff"].notna().sum()), vrp["vrp_diff"].mean(),
             int((~rv["complete_" + ("5m" if rv_col == "rv_5m" else "daily")]).sum()),
             cfg.realized.max_gap_days, int(strips["n_dropped_iv"].sum()), len(cm),
             int(cm["extrapolated"].sum()) if len(cm) else 0,
             int(vrp_cm["vrp_diff"].notna().sum()) if len(cm) else 0, data_dir)
    return tables
