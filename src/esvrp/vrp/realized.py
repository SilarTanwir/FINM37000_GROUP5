"""Realized variance (daily and 5-minute).

Implements PLANNING.md §4.2.

For a chain (one option expiry on one day) the realized variance is measured on the futures contract the
option is written on, over the window from the snapshot to the option's expiry:

    RV = (1/tau) * sum(r_i^2)

with `tau` the chain's own time to expiry in years (ACT/365), the same `tau` the strip uses, so RV and
the variance swap rate are annualized on the same calendar-year basis. Returns come from the bars of
stage 3. Each return runs from one bar's close to the next bar's close, so the move across the
maintenance halt, a weekend or a holiday is one return spanning the gap: nothing is dropped or counted
twice (§4.2, §15.3).

- 5-minute: closes of the traded 5-minute bars. The base is the last bar ending at or before the
  snapshot, and the last return ends at the last bar ending at or before the expiry.
- daily: settlement-to-settlement. The base is the last settlement on or before the snapshot's trade
  date, and the last is the last settlement on or before the expiry's trade date. Days without a
  settlement (shortened holiday sessions) are skipped, which gives one return spanning the gap.
- `kind`: `log` (baseline) or `simple` (Carr-Wu use simple returns; robustness variant).

A window the bars do not cover is not a short window, it is a missing one: summing the returns that exist
and dividing by the full `tau` would understate RV and bias the premium. `rv_table` therefore leaves RV
empty (NaN) when the first or last bar misses its end of the window by more than `max_gap_days`
(default 2: a weekend or a holiday), and says so in `complete_5m` / `complete_daily`. The gaps are kept
in `start_gap_s`, `end_gap_s` (5-minute, seconds) and `start_gap_days`, `end_gap_days` (daily).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from esvrp.data.underlying import SESSIONS, trade_date

FREQS = {"5m": "5m", "1d": "1d", "daily": "1d"}
KINDS = ("log", "simple")
MAX_GAP_DAYS = 2.0  # default tolerance for the bars missing either end of a window (a weekend or holiday)
RV_COLUMNS = [
    "date", "expiry", "underlying", "tau",
    "rv_5m", "rv_5m_simple", "rv_daily", "rv_daily_simple",
    "n_5m", "n_daily", "start_gap_s", "end_gap_s", "start_gap_days", "end_gap_days",
    "complete_5m", "complete_daily",
]


def realized_variance(returns: pd.Series | np.ndarray, tau: float) -> float:
    """RV = (1/tau) * sum(r_i^2), annualized on the same basis as implied variance.

    NaN returns are ignored; NaN is returned if there is no return or `tau` is not positive.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0 or not (np.isfinite(tau) and tau > 0):
        return float("nan")
    return float(np.sum(r**2) / tau)


def _price_series(bars: pd.DataFrame, freq: str, underlying: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Sorted bar times (int64 ns for 5m, datetime64[D] as int64 for daily) and prices."""
    b = bars if underlying is None else bars[bars["underlying"] == underlying]
    if freq == "5m":
        b = b[b["close"].notna()].sort_values("ts")
        t = pd.to_datetime(b["ts"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
        return t, b["close"].to_numpy(dtype=float)
    b = b[b["settle"].notna()].sort_values("date")
    t = pd.to_datetime(b["date"]).dt.as_unit("ns").astype("int64").to_numpy()
    return t, b["settle"].to_numpy(dtype=float)


def _returns(t: np.ndarray, px: np.ndarray, start: int, end: int, kind: str) -> tuple[np.ndarray, int, int]:
    """Returns for the bars after the base bar up to `end`, plus the base and last bar positions.

    The base is the last bar at or before `start`, the last is the last bar at or before `end`. Positions
    are -1 if there is no such bar; there is no return unless the last bar is after the base.
    """
    base = int(np.searchsorted(t, start, side="right")) - 1
    last = int(np.searchsorted(t, end, side="right")) - 1
    if base < 0 or last <= base:
        return np.array([]), base, last
    p = px[base:last + 1]
    r = np.diff(np.log(p)) if kind == "log" else p[1:] / p[:-1] - 1.0
    return r, base, last


def _ns_time(x: Any) -> int:
    ts = pd.Timestamp(x)
    return int((ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")).as_unit("ns").value)


def _ns_date(x: Any) -> int:
    return int(pd.Timestamp(x).as_unit("ns").value)


def returns_from_bars(
    bars: pd.DataFrame, freq: Literal["5m", "1d"], session: str = "full_globex",
    start: Any = None, end: Any = None, underlying: str | None = None, kind: str = "log",
) -> pd.Series:
    """Returns of one underlying contract over a window, one per bar after the base bar.

    `bars` are the stage 3 5-minute bars (`ts` = bar end, UTC) or daily bars (`date`, `settle`). `start`
    and `end` are UTC timestamps for 5-minute bars and dates for daily bars; omit both to use the whole
    series. The result is indexed by the time (or date) of the bar that ends each return. `session` must
    match the one the bars were built with (§15.3); the session filter itself is applied in stage 3.
    """
    if freq not in FREQS:
        raise ValueError(f"freq must be one of {sorted(FREQS)}, got {freq!r}")
    if session not in SESSIONS:
        raise ValueError(f"session must be one of {sorted(SESSIONS)}, got {session!r}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    freq = FREQS[freq]
    # Decided (§4.2, §15.3): full Globex; returns across the maintenance halt, weekends and
    # holidays are kept as one return spanning the gap (gap rule proposed in the README PR).
    t, px = _price_series(bars, freq, underlying)
    empty = pd.Series(dtype=float, name="ret")
    if t.size == 0:
        return empty
    to_ns = _ns_time if freq == "5m" else _ns_date
    lo = t[0] if start is None else to_ns(start)  # no start: the first bar is the base
    hi = t[-1] if end is None else to_ns(end)
    r, base, last = _returns(t, px, lo, hi, kind)
    if r.size == 0:
        return empty
    idx = pd.to_datetime(t[base + 1:last + 1])
    return pd.Series(r, index=idx.tz_localize("UTC") if freq == "5m" else idx.date, name="ret")


def rv_table(
    chains: pd.DataFrame, bars_5m: pd.DataFrame, daily: pd.DataFrame, max_gap_days: float = MAX_GAP_DAYS,
) -> pd.DataFrame:
    """Realized variance for every chain (date, expiry); one row per chain.

    `chains` needs `date`, `expiry`, `underlying`, `snapshot_ts` and `tau` (stage 4, or the strip table
    joined with `snapshot_ts`). Columns: `rv_5m` and `rv_daily` (log returns), their `_simple` variants,
    `n_5m` and `n_daily` (returns used), and the gaps: `start_gap_s` / `end_gap_s`, seconds between the
    snapshot and the base bar and between the last bar and the expiry (5-minute), and `start_gap_days` /
    `end_gap_days` for the daily settlements. If either gap exceeds `max_gap_days` the window is not
    covered by the data: the RV columns of that frequency are NaN and `complete_5m` / `complete_daily`
    is False. Large gaps also mark thin windows or an expiry that does not match the bars (issue #32).
    """
    tol_s = max_gap_days * 86400.0
    c = chains.drop_duplicates(["date", "expiry"]).sort_values(["date", "expiry"]).reset_index(drop=True)
    snap = pd.to_datetime(c["snapshot_ts"], utc=True).dt.as_unit("ns")
    exp = pd.to_datetime(c["expiry"], utc=True).dt.as_unit("ns")
    snap_date = pd.to_datetime(trade_date(snap)).dt.as_unit("ns")
    exp_date = pd.to_datetime(trade_date(exp)).dt.as_unit("ns")
    series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for u in c["underlying"].unique():
        series[(u, "5m")] = _price_series(bars_5m, "5m", u)
        series[(u, "1d")] = _price_series(daily, "1d", u)

    nan = float("nan")
    rows = []
    for i, row in c.iterrows():
        u, tau = row["underlying"], float(row["tau"])
        out = {"date": row["date"], "expiry": row["expiry"], "underlying": u, "tau": tau}
        for freq, lo, hi, tag in (("5m", snap[i].value, exp[i].value, "5m"),
                                  ("1d", snap_date[i].value, exp_date[i].value, "daily")):
            t, px = series[(u, freq)]
            _, base, last = _returns(t, px, lo, hi, "log")
            start_gap = (lo - t[base]) / 1e9 if base >= 0 else nan  # seconds
            end_gap = (hi - t[last]) / 1e9 if last >= 0 else nan
            complete = bool(start_gap <= tol_s and end_gap <= tol_s)  # False if NaN
            if freq == "5m":
                out["start_gap_s"], out["end_gap_s"] = start_gap, end_gap
            else:
                out["start_gap_days"], out["end_gap_days"] = start_gap / 86400, end_gap / 86400
            out[f"complete_{tag}"] = complete
            for kind, suffix in (("log", ""), ("simple", "_simple")):
                r, _, _ = _returns(t, px, lo, hi, kind)
                out[f"rv_{tag}{suffix}"] = realized_variance(r, tau) if complete else nan
                if kind == "log":
                    out[f"n_{tag}"] = int(r.size)
        rows.append(out)
    return pd.DataFrame(rows)[RV_COLUMNS]
