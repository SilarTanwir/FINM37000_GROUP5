"""Tables for the report.

Implements PLANNING.md §6 (Planned outputs 4 and 7) and stage 8: `run` reads the stage 5-7 tables from
`reports/tables/` and writes `reports/tables/headline.csv`, the README's headline numbers in one long
table (one row per hypothesis, source table, row and statistic). An input that is missing is skipped
with a warning, so stage 8 never blocks the final run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from esvrp.data.client import REPO_ROOT
from esvrp.reporting.figures import make_figures

log = logging.getLogger(__name__)

REPORT_DIR = REPO_ROOT / "reports"
HEADLINE_COLUMNS = ["hypothesis", "table", "row", "statistic", "value"]

# Input tables: file name, whether its first column is the index (written with `index=True`).
INPUTS = {"h1": ("h1.csv", False), "summary": ("summary.csv", False), "h3": ("h3.csv", True),
          "strategy": ("strategy.csv", True), "robustness": ("robustness.csv", False)}

H1_STATS = ("n", "mean", "t_naive", "t_hac", "t_hac_30", "sharpe_short")
H2_STATS = ("n", "n_dates", "mean", "t_naive", "t_hac", "t_cluster")
H3_STATS = ("n", "mean_vrp", "t_vrp_hac", "mean_hg", "t_hg_hac", "sign_agree", "pearson", "ci_low",
            "ci_high", "spearman")
H4_ROWS = ("gross_per_vega", "net_per_vega", "net")
H4_STATS = ("n", "mean", "t_hac", "sharpe", "skew", "worst", "max_drawdown", "share_positive")


def _long(df: pd.DataFrame, hypothesis: str, table: str, label_cols: list[str],
          stats: tuple[str, ...] | None = None) -> pd.DataFrame:
    """One row per (input row, statistic); the row label joins `label_cols` with " | "."""
    stats = [c for c in (stats or df.select_dtypes("number").columns) if c in df and c not in label_cols]
    if df.empty or not stats:
        return pd.DataFrame(columns=HEADLINE_COLUMNS)
    labels = df[label_cols].astype(str).agg(" | ".join, axis=1) if label_cols else df.index.astype(str)
    out = df[stats].assign(row=labels.to_numpy()).melt(id_vars="row", var_name="statistic",
                                                       value_name="value")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.assign(hypothesis=hypothesis, table=table)[HEADLINE_COLUMNS].dropna(subset=["value"])


def headline_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Collect the headline numbers from whichever input tables are present."""
    parts = []
    if (h1 := results.get("h1")) is not None:
        parts.append(_long(h1, "H1", "h1", ["sample", "premium"], H1_STATS))
    if (s := results.get("summary")) is not None:
        # Pooled regimes, per unit vega: the headline normalization; the regime rows stay in summary.csv.
        s = s[(s["regime"] == "all") & (s["measure"] == "per_vega")] if {"regime", "measure"} <= set(s) else s
        parts.append(_long(s, "H2", "summary", [c for c in ("moneyness", "maturity") if c in s], H2_STATS))
    if (h3 := results.get("h3")) is not None:
        parts.append(_long(h3, "H3", "h3", [], H3_STATS))
    if (st := results.get("strategy")) is not None:
        parts.append(_long(st.loc[[r for r in H4_ROWS if r in st.index]], "H4", "strategy", [], H4_STATS))
    if (rb := results.get("robustness")) is not None:
        labels = [c for c in rb.columns if not pd.api.types.is_numeric_dtype(rb[c])]
        parts.append(_long(rb, "robustness", "robustness", labels))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=HEADLINE_COLUMNS)


def make_tables(results: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    """Write `headline.csv` to `out_dir` from the input tables in `results`; return the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "headline.csv"
    headline_table(results).to_csv(path, index=False)
    return [path]


def load_inputs(tables_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the stage 5-7 tables that exist; warn about the rest."""
    results = {}
    for name, (file, has_index) in INPUTS.items():
        path = Path(tables_dir) / file
        if not path.exists():
            log.warning("Stage 8: %s not found, its headline rows are skipped", path)
            continue
        results[name] = pd.read_csv(path, index_col=0 if has_index else None)
    return results


def run(cfg: Any, report_dir: Path = REPORT_DIR) -> dict[str, pd.DataFrame]:
    """Stage 8: write `tables/headline.csv` and check the figures under `report_dir`."""
    report_dir = Path(report_dir)
    results = load_inputs(report_dir / "tables")
    make_tables(results, report_dir / "tables")
    figures = make_figures(results, report_dir / "figures")
    headline = pd.read_csv(report_dir / "tables" / "headline.csv")
    log.info("Stage 8: %d headline numbers from %s; %d figures -> %s", len(headline),
             ", ".join(results) or "no tables", len(figures), report_dir)
    return {"headline": headline, **results}
