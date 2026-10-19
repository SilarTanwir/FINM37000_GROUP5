"""Underlying futures bars (5-minute and daily) for the contract that underlies each option.

Implements PLANNING.md §4.2 and §6 stage 3.

Bars are built from `ohlcv-1m` records of the specific futures contracts the options are written on
(the `underlying` of each option, §5.3), never a continuous or front-month series. A 1-minute record is
stamped with its opening time and exists only if the contract traded in that minute. A bar is made only
from minutes that traded, so nothing is filled in while the market is closed or a thin contract is idle:
the move from one bar's close to the next is a single return spanning the gap (§15.3).
Daily settlement prices come from the `statistics` schema.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from esvrp.data.client import DATA_DIR, fetch
from esvrp.data.quotes import TZ
from esvrp.data.universe import check_strikes_bracket_futures

log = logging.getLogger(__name__)

# Session -> (open, close) in Chicago time, or None for every minute traded. Full Globex runs from
# 17:00 CT to 16:00 CT the next day; "rth" is the cash session ending at the snapshot time (§15.4).
SESSIONS: dict[str, tuple[str, str] | None] = {"full_globex": None, "rth": ("08:30", "15:00")}

SESSION_OPEN_CT = "17:00"
SETTLEMENT_STAT = 3  # Databento statistics `stat_type` for the settlement price

BAR_COLUMNS = ["underlying", "date", "ts", "open", "high", "low", "close", "volume"]


def trade_date(ts: pd.Series) -> pd.Series:
    """CME trade date of each UTC time: the session that opens at 17:00 CT belongs to the next day."""
    return (ts.dt.tz_convert(TZ) + pd.Timedelta(hours=7)).dt.date


def _bar_time(raw: pd.DataFrame) -> pd.Series:
    t = raw["ts_event"] if "ts_event" in raw else raw.index.to_series(index=raw.index)
    return pd.to_datetime(t, utc=True)


def build_bars(raw: pd.DataFrame, freq: str, session: str) -> pd.DataFrame:
    """Resample to 5m or daily bars honoring the session definition and maintenance gap.

    One row per contract and bar: `ts` is the end of the bar in UTC (for a daily bar, the end of the
    last minute traded) and `date` the CME trade date. Intervals with no trades give no row.
    """
    # Decided (§4.2, §15.3): full Globex; keep the return across the maintenance halt as one
    # return spanning the gap (gap rule proposed in the README PR).
    if freq not in ("5m", "daily"):
        raise ValueError(f"freq must be '5m' or 'daily', got {freq!r}")
    if session not in SESSIONS:
        raise ValueError(f"session must be one of {sorted(SESSIONS)}, got {session!r}")
    if raw.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    m = raw.assign(underlying=raw["symbol"], start=_bar_time(raw).to_numpy()).sort_values("start")
    if SESSIONS[session]:
        opens, closes = (pd.Timestamp(t).time() for t in SESSIONS[session])
        local = m["start"].dt.tz_convert(TZ).dt.time
        m = m[(local >= opens) & (local < closes)]
    m["date"] = trade_date(m["start"])
    minute = pd.Timedelta(minutes=1)
    m["ts"] = m["start"].dt.floor("5min") + 5 * minute if freq == "5m" else m["start"] + minute
    by = ["underlying", "date", "ts"] if freq == "5m" else ["underlying", "date"]
    bars = m.groupby(by, as_index=False).agg(
        ts=("ts", "last"), open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    )
    return bars[BAR_COLUMNS].sort_values(["underlying", "ts"]).reset_index(drop=True)


def settlements(stats: pd.DataFrame) -> pd.Series:
    """Daily settlement prices per underlying contract, indexed by (underlying, date).

    CME publishes a preliminary settlement about 20 seconds after 15:00 CT and the final one that
    evening, both with `ts_ref` set to the trade date. The last one published wins.
    """
    s = stats[(stats["stat_type"] == SETTLEMENT_STAT) & stats["price"].notna()]
    s = s.assign(underlying=s["symbol"], date=pd.to_datetime(s["ts_ref"], utc=True).dt.date)
    return s.sort_values("ts_event").groupby(["underlying", "date"])["price"].last().rename("settle")


def build_daily(raw: pd.DataFrame, settle: pd.Series, session: str) -> pd.DataFrame:
    """Daily bars with the settlement price attached as `settle`.

    A day with a settlement and no trades (thin back-month contracts) is kept with empty bar fields.
    A day with trades and no settlement (exchange holidays with a shortened session) has no `settle`.
    """
    daily = build_bars(raw, "daily", session)
    daily = daily.merge(settle.reset_index(), on=["underlying", "date"], how="outer")
    daily["volume"] = daily["volume"].fillna(0).astype("int64")
    return daily.sort_values(["underlying", "date"]).reset_index(drop=True)


def contract_windows(universe: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    """Trade dates each underlying contract is needed for: columns underlying, start, end.

    A contract is needed while any option written on it is live, from the first listing to the last
    expiry, clipped to the sample. Contracts with no live option in the sample are left out.
    """
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    g = universe.groupby("underlying").agg(first=("activation", "min"), last=("expiry", "max"))
    first = g["first"].dt.tz_convert(TZ).dt.date.where(g["first"].notna(), start)
    last = g["last"].dt.tz_convert(TZ).dt.date
    w = pd.DataFrame({"underlying": g.index, "start": first.clip(lower=start).to_numpy(),
                      "end": last.clip(upper=end).to_numpy()})
    return w[w["start"] <= w["end"]].reset_index(drop=True)


def _session_open(day: Any) -> pd.Timestamp:
    """When the Globex session for a trade date opens (17:00 CT the evening before), in UTC."""
    prev = pd.Timestamp(day).date() - pd.Timedelta(days=1)
    return pd.Timestamp(f"{prev} {SESSION_OPEN_CT}", tz=TZ).tz_convert("UTC")


def pull_underlying(cfg: Any, universe: pd.DataFrame, client: Any = None,
                    workers: int = 4) -> tuple[pd.DataFrame, pd.Series]:
    """Raw 1-minute bars and settlements for every underlying contract over the days it is needed."""
    windows = contract_windows(universe, cfg.data.start, cfg.data.end)
    if windows.empty:
        raise ValueError("no underlying contracts are live in the sample")
    kw = dict(budget_usd=cfg.data.budget_usd, stype_in="raw_symbol", client=client)

    def pull(w: Any) -> tuple[pd.DataFrame, pd.Series]:
        # Whole trade dates only: from the session open of the first to the session open after the last.
        start, end = _session_open(w.start), _session_open(w.end + pd.Timedelta(days=1))
        bars = fetch(cfg.data.dataset, "ohlcv-1m", w.underlying, start, end, **kw)
        # The final settlement is published after the next session has opened, so read one day further.
        # That also brings in settlements of the days just outside the window; drop those.
        stats = fetch(cfg.data.dataset, "statistics", w.underlying, start, end + pd.Timedelta(days=1), **kw)
        settle = settlements(stats)
        dates = settle.index.get_level_values("date")
        return bars, settle[(dates >= w.start) & (dates <= w.end)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pulled = list(pool.map(pull, windows.itertuples(index=False)))
    bars = [b for b, _ in pulled if not b.empty]
    if not bars:
        raise ValueError("no futures bars found for the underlying contracts in the sample")
    return pd.concat(bars), pd.concat([s for _, s in pulled])


def run(cfg: Any, client: Any = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 3 entry point: `data/underlying_5m.parquet` and `data/underlying_daily.parquet`."""
    universe = pd.read_parquet(DATA_DIR / "universe.parquet")
    raw, settle = pull_underlying(cfg, universe, client=client)
    bars = build_bars(raw, "5m", cfg.realized.session)
    daily = build_daily(raw, settle, cfg.realized.session)
    # Guard against a strike scale error (§5.3): strikes must bracket each contract's last settlement.
    for underlying, price in daily.dropna(subset=["settle"]).groupby("underlying")["settle"].last().items():
        check_strikes_bracket_futures(universe, price, underlying)
    bars.to_parquet(DATA_DIR / "underlying_5m.parquet")
    daily.to_parquet(DATA_DIR / "underlying_daily.parquet")
    log.info("Underlying: %d contracts, %d 5-minute bars, %d contract-days (%d without a settlement, "
             "%d without trades)", daily["underlying"].nunique(), len(bars), len(daily),
             int(daily["settle"].isna().sum()), int(daily["close"].isna().sum()))
    return bars, daily
