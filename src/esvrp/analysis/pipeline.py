"""Stage 7 entry point: the H3 reconciliation and the strategy backtest.

Implements PLANNING.md §6 stage 7 (§4.5-4.7). Reads `vrp.parquet` (stage 5) and `hedged_gains.parquet`
(stage 6) from `data_dir` and writes, under `data_dir`, `h3.parquet` (the joined chains) and, under
`report_dir`:

- `tables/h3.csv`: H3 statistics, every chain and the non-overlapping monthly subsample (roadmap 15)
- `tables/strategy.csv`: risk metrics of the monthly short straddle, gross and net (roadmap 16)
- `tables/strategy_trades.csv`: one row per monthly trade
- `tables/strategy_structures.csv`: the straddle next to strangles at several distances (robustness)
- `tables/strategy_margin.csv`: return on margin at 6%, 8% and 10% of the two legs' notional (sensitivity)
- `tables/strategy_worst_months.csv`: the worst trades with the market move and implied vol behind each
- `figures/h3.png`, `figures/strategy.png`

The moneyness x maturity x regime table (roadmap 14, `tables/summary.csv`) is not written yet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from esvrp.analysis.reconcile import chain_hedged_variance, reconcile
from esvrp.analysis.strategy import margin_sensitivity, run_strategy, run_structures
from esvrp.data.client import DATA_DIR, REPO_ROOT
from esvrp.reporting.figures import h3_figure, strategy_figure

log = logging.getLogger(__name__)

REPORT_DIR = REPO_ROOT / "reports"
MARGIN_FRACTION = 0.10  # assumed margin per contract, a share of futures notional (§4.7: an assumption)


def run(cfg: Any, data_dir: Path = DATA_DIR, report_dir: Path = REPORT_DIR) -> dict[str, pd.DataFrame]:
    """Compute and save the stage 7 tables and figures; return the tables by name."""
    data_dir, report_dir = Path(data_dir), Path(report_dir)
    vrp = pd.read_parquet(data_dir / "vrp.parquet")
    gains = pd.read_parquet(data_dir / "hedged_gains.parquet")

    joined, h3 = reconcile(vrp, chain_hedged_variance(gains, "iv_fixed"))
    strategy = run_strategy(gains, cfg.hedging.hedge_delta, MARGIN_FRACTION)
    structures = run_structures(gains, cfg.hedging.hedge_delta, MARGIN_FRACTION)
    margin = margin_sensitivity(strategy.trades)

    tables_dir, figures_dir = report_dir / "tables", report_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(data_dir / "h3.parquet")
    h3.to_csv(tables_dir / "h3.csv")
    strategy.metrics.to_csv(tables_dir / "strategy.csv")
    strategy.trades.to_csv(tables_dir / "strategy_trades.csv", index=False)
    structures.to_csv(tables_dir / "strategy_structures.csv")
    margin.to_csv(tables_dir / "strategy_margin.csv")
    strategy.worst.to_csv(tables_dir / "strategy_worst_months.csv", index=False)
    h3_figure(joined, h3, figures_dir / "h3.png")
    strategy_figure(strategy.trades, strategy.metrics, figures_dir / "strategy.png")

    d = h3.iloc[0]
    net = strategy.metrics.loc["net_per_vega"]
    log.info("Stage 7: H3 on %d chains, correlation %.2f (%.2f to %.2f); strategy %d trades, net %.2f vol "
             "points per unit vega, Sharpe %.2f -> %s", int(d.get("n", 0)), d.get("pearson", float("nan")),
             d.get("ci_low", float("nan")), d.get("ci_high", float("nan")), int(net.get("n", 0)),
             net.get("mean", float("nan")), net.get("sharpe", float("nan")), report_dir)
    return {"h3_chains": joined, "h3": h3, "strategy": strategy.metrics, "strategy_trades": strategy.trades,
            "strategy_structures": structures, "strategy_worst_months": strategy.worst,
            "strategy_margin": margin}
