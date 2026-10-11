"""Chain assembly: forward, time to expiry, mid, implied volatility, Greeks and moneyness per quote.

Implements PLANNING.md §6 stage 4 and §4.1 (Forward price, Implied volatility).

Each cleaned quote from stage 2 gets the forward of its chain (one expiry on one day). The forward is
estimated from the option quotes themselves by put-call parity, because the option and futures quotes are
not perfectly synchronous and a small error in F moves every implied volatility in the chain. The futures
mid at the snapshot is kept next to it, and the difference between the two is the stage's diagnostic.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from esvrp.data.client import DATA_DIR, REPO_ROOT
from esvrp.data.rates import risk_free_rate
from esvrp.pricing.black76 import greeks, implied_vols
from esvrp.pricing.forward import parity_forward

log = logging.getLogger(__name__)

FIGURE_PATH = REPO_ROOT / "reports" / "figures" / "forward_gap.png"

CHAIN_COLUMNS = ["tau", "r", "mid", "F_fut", "F_parity", "F", "forward_gap", "settle", "moneyness",
                 "iv", "delta", "gamma", "vega"]


def build_chain(quotes: pd.DataFrame, r: Any, settle: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add the chain columns to cleaned quotes; one row per quote, none dropped.

    `r` is a constant or a Series of rates indexed by date. `settle` is the stage 3 daily table; its
    settlement price is attached per (date, underlying) for reference.

    - `tau`: years from the snapshot to expiry (ACT/365).
    - `mid`: bid/ask midpoint; empty for one-sided quotes, which therefore get no implied volatility.
    - `F_fut`: futures mid of the option's underlying at the snapshot (the `F` of stage 2).
    - `F_parity`: parity-implied forward of the chain; `forward_gap` = `F_parity` - `F_fut`.
    - `F`: the forward used for pricing: `F_parity`, or `F_fut` if no strike has a two-sided call and put.
    - `moneyness`: ln(K / F) (§4.5). `iv`, `delta`, `gamma`, `vega`: Black-76 at the option's own mid.
    """
    c = quotes.rename(columns={"F": "F_fut"}).reset_index(drop=True)
    c["tau"] = ((pd.to_datetime(c["expiry"], utc=True) - pd.to_datetime(c["snapshot_ts"], utc=True))
                .dt.total_seconds() / (365 * 86400))
    c["r"] = c["date"].map(r) if isinstance(r, pd.Series) else float(r)
    one_sided = c["one_sided"] if "one_sided" in c else c["ask"].isna()
    c["mid"] = ((c["bid"] + c["ask"]) / 2).where(~one_sided.astype(bool))

    by = ["date", "expiry"]
    forwards = [(*key, parity_forward(g, g["r"].iloc[0], g["tau"].iloc[0])) for key, g in c.groupby(by)]
    c = c.merge(pd.DataFrame(forwards, columns=[*by, "F_parity"]), on=by, how="left")
    c["F"] = c["F_parity"].fillna(c["F_fut"])
    c["forward_gap"] = c["F_parity"] - c["F_fut"]
    if settle is not None:
        c = c.merge(settle[["date", "underlying", "settle"]], on=["date", "underlying"], how="left")
    else:
        c["settle"] = np.nan

    c["moneyness"] = np.log(c["strike"] / c["F"])
    c["iv"] = implied_vols(c["mid"], c["F"], c["strike"], c["tau"], c["r"], c["cp"])
    for name, value in greeks(c["F"], c["strike"], c["tau"], c["r"], c["iv"], c["cp"]).items():
        c[name] = value
    return c[[*(col for col in c.columns if col not in CHAIN_COLUMNS), *CHAIN_COLUMNS]]


def forward_gap_table(chain: pd.DataFrame) -> pd.DataFrame:
    """One row per chain (date, expiry): underlying, tau, F_fut, F_parity and forward_gap."""
    cols = ["date", "expiry", "underlying", "tau", "F_fut", "F_parity", "forward_gap"]
    return chain[cols].drop_duplicates(["date", "expiry"]).reset_index(drop=True)


def run(cfg: Any) -> pd.DataFrame:
    """Stage 4 entry point: `data/chain.parquet` and the forward-gap diagnostic figure."""
    from esvrp.reporting.figures import forward_gap_figure

    quotes = pd.read_parquet(DATA_DIR / "quotes.parquet")
    daily = pd.read_parquet(DATA_DIR / "underlying_daily.parquet")
    dates = pd.Index(sorted(quotes["date"].unique()))
    r = risk_free_rate(dates, cfg.pricing.rate_source, cfg.pricing.rate_constant)
    chain = build_chain(quotes, r, daily)
    chain.to_parquet(DATA_DIR / "chain.parquet")
    gaps = forward_gap_table(chain)
    forward_gap_figure(gaps, FIGURE_PATH)
    two_sided = chain["mid"].notna()
    log.info("Chain: %d quotes in %d chains; %d chains without a parity forward; %d two-sided quotes "
             "without an implied vol; median |forward gap| %.2f points -> %s",
             len(chain), len(gaps), int(gaps["F_parity"].isna().sum()),
             int((two_sided & chain["iv"].isna()).sum()), gaps["forward_gap"].abs().median(), FIGURE_PATH)
    return chain
