"""Daily quote snapshot loader and cleaning rules.

Implements PLANNING.md §6 stage 2 (cleaning rules) and §5.3.

Each trading day we keep the last `bbo-1m` record at or before the snapshot time (15:00 CT, §15.4) for
every option in the universe. A `bbo-1m` record stamped T carries the best bid/offer as of T. Options
with no record in the snapshot minute are back-filled from the most recent record within
`cleaning.backfill_minutes`; older quotes are treated as missing. The futures mid of each option's
underlying contract at the same minute is attached as `F` for the cleaning rules that need it.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from esvrp.data.client import DATA_DIR, fetch
from esvrp.data.universe import SERIES_PARENT

log = logging.getLogger(__name__)

TZ = "America/Chicago"

QUOTE_COLUMNS = [
    "date", "snapshot_ts", "instrument_id", "raw_symbol", "cp", "strike", "expiry", "underlying",
    "user_defined", "bid", "ask", "bid_sz", "ask_sz", "quote_ts", "quote_age_s",
]


@dataclass(frozen=True)
class _DefaultCleaning:
    min_strikes: int = 10
    min_range_pct: float = 0.10
    drop_zero_bid: bool = True
    drop_crossed: bool = True
    backfill_minutes: int = 15
    rate: float = 0.04


def snapshot_ts(day: Any, snapshot_time_ct: str) -> pd.Timestamp:
    """The snapshot instant for a trading day, in UTC (handles CST/CDT)."""
    return pd.Timestamp(f"{pd.Timestamp(day).date()} {snapshot_time_ct}", tz=TZ).tz_convert("UTC")


def _quote_time(bbo: pd.DataFrame) -> pd.Series:
    t = bbo["ts_recv"] if "ts_recv" in bbo else bbo.index.to_series(index=bbo.index)
    return pd.to_datetime(t, utc=True)


def load_snapshots(
    universe: pd.DataFrame, bbo: pd.DataFrame, snapshot_time_ct: str, backfill_minutes: int = 15,
) -> pd.DataFrame:
    """Last bid/ask per option before the snapshot time, one row per option-day."""
    # Decided (§15.4): 15:00 CT. Back-fill strikes with no quote at that minute (§5.2).
    if bbo.empty:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    q = bbo.assign(quote_ts=_quote_time(bbo).to_numpy())
    q["date"] = q["quote_ts"].dt.tz_convert(TZ).dt.date
    snaps = {d: snapshot_ts(d, snapshot_time_ct) for d in q["date"].unique()}
    q["snapshot_ts"] = q["date"].map(snaps)
    window = pd.Timedelta(minutes=backfill_minutes)
    q = q[(q["quote_ts"] <= q["snapshot_ts"]) & (q["quote_ts"] > q["snapshot_ts"] - window)]
    last = q.sort_values("quote_ts").groupby(["date", "instrument_id"]).tail(1)

    out = last.merge(universe, on="instrument_id", how="inner")
    out = out[out["expiry"] > out["snapshot_ts"]]  # an option expiring at the snapshot is no longer live
    if out.empty:  # e.g. holidays with an early close: nothing quoted near the snapshot
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    out = out.rename(columns={"bid_px_00": "bid", "ask_px_00": "ask", "bid_sz_00": "bid_sz",
                              "ask_sz_00": "ask_sz"})
    out["quote_age_s"] = (out["snapshot_ts"] - out["quote_ts"]).dt.total_seconds()
    return out[QUOTE_COLUMNS].sort_values(["date", "expiry", "cp", "strike"]).reset_index(drop=True)


def futures_mid_at_snapshot(futures_bbo: pd.DataFrame, snapshot_time_ct: str,
                            backfill_minutes: int = 15) -> pd.DataFrame:
    """Mid of each futures contract at the snapshot: columns date, underlying, F."""
    if futures_bbo.empty:
        return pd.DataFrame(columns=["date", "underlying", "F"])
    f = futures_bbo.assign(quote_ts=_quote_time(futures_bbo).to_numpy())
    f["date"] = f["quote_ts"].dt.tz_convert(TZ).dt.date
    f["snapshot_ts"] = f["date"].map({d: snapshot_ts(d, snapshot_time_ct) for d in f["date"].unique()})
    window = pd.Timedelta(minutes=backfill_minutes)
    f = f[(f["quote_ts"] <= f["snapshot_ts"]) & (f["quote_ts"] > f["snapshot_ts"] - window)]
    f = f.dropna(subset=["bid_px_00", "ask_px_00"])
    if f.empty:
        return pd.DataFrame(columns=["date", "underlying", "F"])
    last = f.sort_values("quote_ts").groupby(["date", "symbol"]).tail(1)
    return pd.DataFrame({"date": last["date"], "underlying": last["symbol"],
                         "F": (last["bid_px_00"] + last["ask_px_00"]) / 2}).reset_index(drop=True)


def clean_quotes(quotes: pd.DataFrame, cfg: object) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply cleaning rules; return cleaned quotes and a per-rule drop count log.

    Row rules, in order: no quote on either side, zero or missing bid, crossed, locked, no futures price
    for the underlying, mid below discounted intrinsic value. A missing ask with a valid bid is flagged
    `one_sided` and kept.
    Chain rules per (date, expiry): at least `min_strikes` strikes, and strikes spanning at least
    F(1 - min_range_pct) to F(1 + min_range_pct). Rules that need F are skipped if there is no `F` column.
    """
    c = cfg or _DefaultCleaning()
    rate = getattr(c, "rate", 0.04)
    q = quotes.copy()
    counts: dict[str, int] = {}

    def drop(name: str, mask: pd.Series) -> None:
        nonlocal q
        counts[name] = int(mask.sum())
        q = q[~mask]

    no_bid = q["bid"].isna() | (q["bid"] <= 0)
    no_ask = q["ask"].isna() | (q["ask"] <= 0)
    drop("no_quote", no_bid & no_ask)
    no_bid = q["bid"].isna() | (q["bid"] <= 0)
    drop("zero_bid", no_bid if c.drop_zero_bid else no_bid & False)
    q["one_sided"] = q["ask"].isna() | (q["ask"] <= 0)
    two_sided = ~q["one_sided"]
    drop("crossed", two_sided & (q["bid"] > q["ask"]) if c.drop_crossed else two_sided & False)
    two_sided = ~q["one_sided"]
    drop("locked", two_sided & (q["bid"] == q["ask"]) if c.drop_crossed else two_sided & False)

    has_f = "F" in q
    if has_f:
        # Back-month futures (e.g. ESZ7, ESH8 in Sep 2026) can go unquoted near the snapshot.
        drop("no_futures_price", q["F"].isna())
        tau = ((pd.to_datetime(q["expiry"], utc=True) - pd.to_datetime(q["snapshot_ts"], utc=True))
               .dt.total_seconds() / (365 * 86400))
        payoff = np.where(q["cp"] == "c", q["F"] - q["strike"], q["strike"] - q["F"])
        intrinsic = np.exp(-rate * tau) * np.maximum(payoff, 0.0)
        mid = np.where(q["one_sided"], q["bid"], (q["bid"] + q["ask"]) / 2)
        drop("below_intrinsic", pd.Series(mid < intrinsic - 1e-9, index=q.index))
    else:
        counts["no_futures_price"] = counts["below_intrinsic"] = 0
        log.warning("No F column: intrinsic-value and strike-range rules skipped")

    has_chain = {"date", "expiry"} <= set(q.columns)
    if has_chain:
        chain = q.groupby(["date", "expiry"])["strike"]
        drop("chain_min_strikes", chain.transform("nunique") < c.min_strikes)
    else:
        counts["chain_min_strikes"] = 0
    if has_f and has_chain:
        chain = q.groupby(["date", "expiry"])
        lo_ok = chain["strike"].transform("min") <= q["F"] * (1 - c.min_range_pct)
        hi_ok = chain["strike"].transform("max") >= q["F"] * (1 + c.min_range_pct)
        drop("chain_min_range", ~(lo_ok & hi_ok))
    else:
        counts["chain_min_range"] = 0

    counts["kept"] = len(q)
    counts["one_sided_kept"] = int(q["one_sided"].sum())
    log.info("Cleaning: %s", counts)
    return q.reset_index(drop=True), counts


def _pull_day(cfg: Any, universe: pd.DataFrame, day: pd.Timestamp, client: Any) -> pd.DataFrame:
    snap = snapshot_ts(day, cfg.data.snapshot_time_ct)
    start = snap - pd.Timedelta(minutes=cfg.cleaning.backfill_minutes)
    end = snap + pd.Timedelta(minutes=1)
    kw = dict(budget_usd=cfg.data.budget_usd, stype_in="parent", client=client)
    opts = fetch(cfg.data.dataset, "bbo-1m", SERIES_PARENT[cfg.data.series], start, end, **kw)
    if opts.empty:
        return pd.DataFrame()
    quotes = load_snapshots(universe, opts, cfg.data.snapshot_time_ct, cfg.cleaning.backfill_minutes)
    if quotes.empty:
        return quotes
    # Only the futures the live options are written on; the ES.FUT parent also carries every spread.
    underlyings = sorted(quotes["underlying"].unique())
    futs = fetch(cfg.data.dataset, "bbo-1m", underlyings, start, end,
                 budget_usd=cfg.data.budget_usd, stype_in="raw_symbol", client=client)
    fmid = futures_mid_at_snapshot(futs, cfg.data.snapshot_time_ct, cfg.cleaning.backfill_minutes)
    return quotes.merge(fmid, on=["date", "underlying"], how="left")


def pull_snapshots(cfg: Any, universe: pd.DataFrame, client: Any = None, workers: int = 4) -> pd.DataFrame:
    """Raw (uncleaned) snapshots for every business day in the sample, with F attached."""
    days = pd.bdate_range(cfg.data.start, cfg.data.end)
    failed: list[str] = []

    def pull(day: pd.Timestamp) -> pd.DataFrame:
        try:
            return _pull_day(cfg, universe, day, client)
        except Exception as err:  # keep going; finished days are cached, so a rerun only redoes these
            log.error("Day %s failed after retries: %s", day.date(), err)
            failed.append(str(day.date()))
            return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(pull, days))
    if failed:
        raise RuntimeError(f"{len(failed)} day(s) failed; rerun stage 2 to retry them: {sorted(failed)}")
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("no option quotes found in the sample")
    return pd.concat(frames, ignore_index=True)


def run(cfg: Any, client: Any = None) -> pd.DataFrame:
    """Stage 2 entry point: `data/quotes.parquet` plus a per-rule cleaning log."""
    universe = pd.read_parquet(DATA_DIR / "universe.parquet")
    raw = pull_snapshots(cfg, universe, client=client)
    clean_cfg = _DefaultCleaning(**{**vars(cfg.cleaning), "rate": cfg.pricing.rate_constant})
    cleaned, counts = clean_quotes(raw, clean_cfg)
    cleaned.to_parquet(DATA_DIR / "quotes.parquet")
    pd.Series(counts, name="rows").to_csv(DATA_DIR / "quotes_cleaning_log.csv")
    log.info("Quotes: %d option-days over %d days", len(cleaned), cleaned["date"].nunique())
    return cleaned
