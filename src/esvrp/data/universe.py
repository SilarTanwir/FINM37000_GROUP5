"""Option universe from Databento definition records.

Implements PLANNING.md §5 and §6 stage 1. Always use the `underlying` from the definition (§5.3).

Definitions for the `EW.OPT` parent contain single options (instrument_class C/P) and spreads
(T = exchange-listed strategies, M = user-defined multi-leg). Only single options are kept.
Exercise style comes from the CFI code: character 3 is E (European) or A (American).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from esvrp.config import Config
from esvrp.data.client import DATA_DIR, fetch

log = logging.getLogger(__name__)

# Option parent symbol per series (§5.3). End-of-month ES options are listed under EW, not ES.
SERIES_PARENT = {"monthly_eom": "EW.OPT"}

STYLE_FROM_CFI = {"E": "european", "A": "american", "B": "bermudan"}

UNIVERSE_COLUMNS = [
    "instrument_id", "raw_symbol", "cp", "strike", "expiry", "underlying", "style",
    "user_defined", "activation",
]


def build_universe(definitions: pd.DataFrame, european_only: bool = True) -> pd.DataFrame:
    """One row per option: strike, expiry, cp, underlying, exercise style.

    Keeps calls and puts only, one row per instrument (latest definition wins), and by default only
    European exercise (the monthly_eom series, §4.1).
    """
    d = definitions[definitions["instrument_class"].isin(["C", "P"])].copy()
    if "instrument_id" not in d:
        d["instrument_id"] = range(len(d))
    d = d.drop_duplicates("instrument_id", keep="last")

    cfi = d["cfi"] if "cfi" in d else pd.Series(pd.NA, index=d.index)
    uni = pd.DataFrame({
        "instrument_id": d["instrument_id"].astype("int64"),
        "raw_symbol": d["raw_symbol"],
        "cp": d["instrument_class"].str.lower(),
        "strike": d["strike_price"].astype(float),
        "expiry": pd.to_datetime(d["expiration"], utc=True),
        "underlying": d["underlying"],
        "style": cfi.str[2].map(STYLE_FROM_CFI),
        "user_defined": d.get("user_defined_instrument", pd.Series("N", index=d.index)).eq("Y"),
        "activation": pd.to_datetime(d.get("activation", pd.Series(pd.NaT, index=d.index)), utc=True),
    })
    if european_only:
        dropped = uni["style"].ne("european")
        if dropped.any():
            log.info("Dropping %d non-European options", int(dropped.sum()))
        uni = uni[~dropped]
    if (uni["underlying"].fillna("") == "").any():
        raise ValueError("option definitions with no underlying future; check the definition records")
    return uni.sort_values(["expiry", "cp", "strike"]).reset_index(drop=True)[UNIVERSE_COLUMNS]


def check_strikes_bracket_futures(
    universe: pd.DataFrame, futures_price: float, underlying: str | None = None,
) -> None:
    """Sanity check that strikes bracket the futures price (display-factor issue, §5.3).

    A strike scale error (e.g. strikes off by a power of ten) puts every strike on one side of F.
    Pass `underlying` to check only the options written on that futures contract.
    """
    # ES is not on Databento's list of symbols affected by the strike_price display-factor
    # issue (§5.3); this check stays as a guard.
    rows = universe["underlying"] == underlying if underlying else slice(None)
    strikes = universe.loc[rows, "strike"]
    if strikes.empty:
        raise ValueError(f"no strikes to check for underlying {underlying!r}")
    lo, hi = strikes.min(), strikes.max()
    if not lo < futures_price < hi:
        raise ValueError(
            f"strikes [{lo}, {hi}] do not bracket futures price {futures_price} "
            f"(underlying {underlying or 'all'}); possible strike scale error"
        )


def snapshot_dates(start: Any, end: Any, freq: str = "W-TUE") -> list[pd.Timestamp]:
    """Dates on which to read the definition snapshot: weekly by default.

    Weekly snapshots catch strikes CME adds as the market moves, which a single monthly snapshot misses.
    """
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=freq))


def pull_universe(cfg: Config, client: Any = None, freq: str = "W-TUE", workers: int = 4) -> pd.DataFrame:
    """Union of the option universe across weekly definition snapshots over the sample (stage 1)."""
    parent = SERIES_PARENT[cfg.data.series]

    def pull(day: pd.Timestamp) -> pd.DataFrame:
        return fetch(cfg.data.dataset, "definition", parent, day.date(), (day + pd.Timedelta(days=1)).date(),
                     budget_usd=cfg.data.budget_usd, stype_in="parent", client=client)

    with ThreadPoolExecutor(max_workers=workers) as pool:  # map keeps date order for "latest wins"
        snapshots = list(pool.map(pull, snapshot_dates(cfg.data.start, cfg.data.end, freq)))
    frames = [build_universe(defs) for defs in snapshots if not defs.empty]
    if not frames:
        raise ValueError(f"no {parent} definitions found between {cfg.data.start} and {cfg.data.end}")
    uni = pd.concat(frames).drop_duplicates("instrument_id", keep="last")
    return uni.sort_values(["expiry", "cp", "strike"]).reset_index(drop=True)


def run(cfg: Config, client: Any = None) -> pd.DataFrame:
    """Stage 1 entry point: build and save `data/universe.parquet`."""
    uni = pull_universe(cfg, client=client)
    out = DATA_DIR / "universe.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    uni.to_parquet(out)
    log.info("Universe: %d options, %d expiries -> %s", len(uni), uni["expiry"].nunique(), out)
    return uni
