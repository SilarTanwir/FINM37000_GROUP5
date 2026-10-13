"""Tests for data/underlying.py (PLANNING.md §6 stage 3, roadmap 6). Fixtures only, no network."""

import numpy as np
import pandas as pd
import pytest

from esvrp.data.underlying import (
    build_bars,
    build_daily,
    contract_windows,
    pull_underlying,
    settlements,
    trade_date,
)


def minutes(rows, symbol="ESZ6"):
    """ohlcv-1m records: (utc open time of the minute, close). Open = high = low = close, volume 1."""
    df = pd.DataFrame(rows, columns=["ts_event", "close"])
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df["open"] = df["high"] = df["low"] = df["close"]
    df["volume"] = 1
    df["symbol"] = symbol
    return df.set_index("ts_event")


def stats(rows, symbol="ESZ6"):
    """statistics records: (utc publication time, trade date, stat_type, price)."""
    df = pd.DataFrame(rows, columns=["ts_event", "ts_ref", "stat_type", "price"])
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df["ts_ref"] = pd.to_datetime(df["ts_ref"], utc=True)
    df["symbol"] = symbol
    return df


def day(s):
    return pd.Timestamp(s).date()


def test_trade_date_rolls_at_session_open():
    ts = pd.Series(pd.to_datetime([
        "2026-09-01 20:59",   # Tue 15:59 CDT
        "2026-09-01 22:00",   # Tue 17:00 CDT: Wednesday's session opens
        "2026-09-06 22:00",   # Sun 17:00 CDT: Monday's session opens
        "2026-12-01 22:59",   # Tue 16:59 CST
        "2026-12-01 23:00",   # Tue 17:00 CST
    ], utc=True))
    assert list(trade_date(ts)) == [day("2026-09-01"), day("2026-09-02"), day("2026-09-07"),
                                    day("2026-12-01"), day("2026-12-02")]


def test_five_minute_bars_are_stamped_at_their_end():
    raw = pd.DataFrame({
        "ts_event": pd.to_datetime(["2026-09-01 19:55", "2026-09-01 19:57", "2026-09-01 19:59",
                                    "2026-09-01 20:00"], utc=True),
        "open": [100.0, 102.0, 101.0, 105.0], "high": [103.0, 104.0, 101.5, 106.0],
        "low": [99.0, 101.0, 98.0, 105.0], "close": [102.0, 101.0, 99.5, 106.0],
        "volume": [10, 20, 30, 40], "symbol": "ESZ6",
    }).set_index("ts_event")
    bars = build_bars(raw, "5m", "full_globex")
    assert list(bars["ts"]) == [pd.Timestamp("2026-09-01 20:00", tz="UTC"),    # 14:55-15:00 CT
                                pd.Timestamp("2026-09-01 20:05", tz="UTC")]    # the 15:00 minute is later
    first = bars.iloc[0]
    assert (first["open"], first["high"], first["low"], first["close"], first["volume"]) == \
        (100.0, 104.0, 98.0, 99.5, 60)
    assert first["underlying"] == "ESZ6" and first["date"] == day("2026-09-01")


def test_gap_is_one_return_and_no_bars_are_invented():
    raw = minutes([
        ("2026-09-04 20:54", 100.0),   # Fri 15:54 CDT
        ("2026-09-04 20:59", 101.0),   # Fri 15:59: last minute before the weekend
        ("2026-09-06 22:00", 104.0),   # Sun 17:00: reopen
        ("2026-09-06 22:07", 103.0),
        ("2026-09-07 20:59", 105.0),   # Mon 15:59: last minute before the maintenance halt
        ("2026-09-07 22:02", 102.0),   # Mon 17:02: after the halt
    ])
    bars = build_bars(raw, "5m", "full_globex")
    assert len(bars) == 6                                          # one per traded interval, none filled
    assert list(bars["close"]) == [100.0, 101.0, 104.0, 103.0, 105.0, 102.0]
    gaps = bars["ts"].diff().dt.total_seconds().iloc[1:] / 3600
    assert gaps.max() == pytest.approx(49 + 5 / 60)                # the weekend sits inside one return
    returns = np.log(bars["close"]).diff().dropna()
    assert len(returns) == 5
    assert returns.iloc[1] == pytest.approx(np.log(104.0 / 101.0))  # across the weekend
    assert returns.iloc[4] == pytest.approx(np.log(102.0 / 105.0))  # across the halt
    assert returns.sum() == pytest.approx(np.log(102.0 / 100.0))    # nothing dropped or counted twice
    assert list(bars["date"]) == [day("2026-09-04")] * 2 + [day("2026-09-07")] * 3 + [day("2026-09-08")]


def test_rth_session_keeps_cash_hours_only():
    raw = minutes([
        ("2026-09-01 13:29", 99.0),    # 08:29 CDT: before the open
        ("2026-09-01 13:30", 100.0),   # 08:30: first RTH minute
        ("2026-09-01 19:59", 101.0),   # 14:59: last RTH minute
        ("2026-09-01 20:00", 102.0),   # 15:00: after the close
    ])
    assert list(build_bars(raw, "5m", "rth")["close"]) == [100.0, 101.0]
    rth = build_bars(raw, "daily", "rth").iloc[0]
    assert (rth["open"], rth["close"]) == (100.0, 101.0)
    full = build_bars(raw, "daily", "full_globex").iloc[0]
    assert (full["open"], full["close"], full["volume"]) == (99.0, 102.0, 4)


def test_daily_bar_spans_the_evening_and_day_session():
    raw = minutes([
        ("2026-08-31 22:00", 100.0),   # Mon 17:00 CDT: Tuesday's session opens
        ("2026-09-01 14:00", 110.0),
        ("2026-09-01 20:59", 105.0),   # Tue 15:59: last minute of Tuesday's session
        ("2026-09-01 22:00", 106.0),   # Tue 17:00: Wednesday's session
    ])
    daily = build_bars(raw, "daily", "full_globex")
    assert list(daily["date"]) == [day("2026-09-01"), day("2026-09-02")]
    tue = daily.iloc[0]
    assert (tue["open"], tue["high"], tue["low"], tue["close"]) == (100.0, 110.0, 100.0, 105.0)
    assert tue["volume"] == 3
    assert tue["ts"] == pd.Timestamp("2026-09-01 21:00", tz="UTC")


def test_contracts_are_kept_separate():
    raw = pd.concat([minutes([("2026-09-01 20:00", 100.0), ("2026-09-01 20:01", 101.0)], "ESZ6"),
                     minutes([("2026-09-01 20:01", 200.0)], "ESH7")])
    bars = build_bars(raw, "5m", "full_globex")
    assert bars.set_index("underlying")["close"].to_dict() == {"ESH7": 200.0, "ESZ6": 101.0}


def test_last_published_settlement_wins():
    s = settlements(stats([
        ("2026-09-01 20:00:20", "2026-09-01", 3, 5000.00),   # preliminary
        ("2026-09-01 23:07:11", "2026-09-01", 3, 5000.25),   # final
        ("2026-09-01 20:00:20", "2026-09-01", 9, 123456.0),  # another statistic: ignored
        ("2026-09-02 20:00:20", "2026-09-02", 3, 5010.00),
        ("2026-09-03 20:00:20", "2026-09-03", 3, float("nan")),
    ]))
    assert s.to_dict() == {("ESZ6", day("2026-09-01")): 5000.25, ("ESZ6", day("2026-09-02")): 5010.0}


def test_daily_bars_with_settlements():
    raw = minutes([("2026-09-01 14:00", 5001.0), ("2026-09-07 14:00", 5020.0)])   # Sep 7: Labor Day
    settle = settlements(stats([("2026-09-01 20:00:20", "2026-09-01", 3, 5000.0),
                                ("2026-09-02 20:00:20", "2026-09-02", 3, 5010.0)]))
    daily = build_daily(raw, settle, "full_globex").set_index("date")
    assert list(daily.index) == [day("2026-09-01"), day("2026-09-02"), day("2026-09-07")]
    assert (daily.loc[day("2026-09-01"), "close"], daily.loc[day("2026-09-01"), "settle"]) == (5001.0, 5000.0)
    assert np.isnan(daily.loc[day("2026-09-02"), "close"])       # settled but did not trade: kept
    assert (daily.loc[day("2026-09-02"), "settle"], daily.loc[day("2026-09-02"), "volume"]) == (5010.0, 0)
    assert np.isnan(daily.loc[day("2026-09-07"), "settle"])      # holiday session: traded, no settlement


def test_empty_and_invalid_inputs():
    assert build_bars(minutes([]), "5m", "full_globex").empty
    with pytest.raises(ValueError, match="freq"):
        build_bars(minutes([]), "1h", "full_globex")
    with pytest.raises(ValueError, match="session"):
        build_bars(minutes([]), "5m", "eth")


UNIVERSE = pd.DataFrame({
    "underlying": ["ESU6", "ESZ6", "ESZ6", "ESH7", "ESM8"],
    "activation": pd.to_datetime(["2025-01-02 21:25", "2026-08-10 20:25", "2026-09-14 20:25",
                                  None, "2026-10-05 20:25"], utc=True),
    "expiry": pd.to_datetime(["2026-08-31 20:00", "2026-09-30 20:00", "2026-11-30 21:00",
                              "2026-12-31 21:00", "2028-03-31 20:00"], utc=True),
})


def test_contract_windows_follow_the_options_and_the_sample():
    w = contract_windows(UNIVERSE, "2026-09-01", "2026-09-30").set_index("underlying")
    assert list(w.index) == ["ESH7", "ESZ6"]                      # ESU6 expired, ESM8 not listed yet
    assert (w.loc["ESZ6", "start"], w.loc["ESZ6", "end"]) == (day("2026-09-01"), day("2026-09-30"))
    assert w.loc["ESH7", "start"] == day("2026-09-01")            # unknown listing date: sample start
    w = contract_windows(UNIVERSE, "2026-08-01", "2026-12-31", tail_days=0).set_index("underlying")
    assert (w.loc["ESZ6", "start"], w.loc["ESZ6", "end"]) == (day("2026-08-10"), day("2026-11-30"))


def test_contract_windows_run_past_the_last_expiry_for_realized_variance():
    # A 30-day realized-variance window from a late-August snapshot needs ESU6 bars into September.
    w = contract_windows(UNIVERSE, "2026-08-01", "2026-12-31").set_index("underlying")
    assert w.loc["ESU6", "end"] == day("2026-09-30")              # last option expiry 31 Aug + 30 days
    assert w.loc["ESZ6", "end"] == day("2026-12-30")
    assert w.loc["ESH7", "end"] == day("2026-12-31")              # clipped to the sample
    # The tail does not bring back a contract whose options all expired before the sample.
    assert "ESU6" not in set(contract_windows(UNIVERSE, "2026-09-01", "2026-09-30")["underlying"])


def test_pull_underlying_requests_each_contract_for_whole_trade_dates(monkeypatch):
    from types import SimpleNamespace

    import esvrp.data.underlying as underlying

    calls = []

    def fake_fetch(dataset, schema, symbols, start, end, budget_usd, stype_in, client):
        calls.append((schema, symbols, stype_in, str(start), str(end)))
        if schema == "statistics":
            return stats([("2026-08-30 13:00:00", "2026-08-28", 3, 4990.0),   # Friday's, republished
                          ("2026-09-01 20:00:20", "2026-09-01", 3, 5000.0),
                          ("2026-10-01 20:00:20", "2026-10-01", 3, 5100.0)], symbols)
        return minutes([("2026-09-01 14:00", 5001.0)], symbols) if symbols == "ESZ6" else minutes([])

    monkeypatch.setattr(underlying, "fetch", fake_fetch)
    cfg = SimpleNamespace(data=SimpleNamespace(dataset="GLBX.MDP3", start="2026-09-01", end="2026-09-30",
                                               budget_usd=50))
    raw, settle = pull_underlying(cfg, UNIVERSE)
    assert sorted(calls) == [
        ("ohlcv-1m", "ESH7", "raw_symbol", "2026-08-31 22:00:00+00:00", "2026-09-30 22:00:00+00:00"),
        ("ohlcv-1m", "ESZ6", "raw_symbol", "2026-08-31 22:00:00+00:00", "2026-09-30 22:00:00+00:00"),
        ("statistics", "ESH7", "raw_symbol", "2026-08-31 22:00:00+00:00", "2026-10-01 22:00:00+00:00"),
        ("statistics", "ESZ6", "raw_symbol", "2026-08-31 22:00:00+00:00", "2026-10-01 22:00:00+00:00"),
    ]
    assert list(raw["symbol"]) == ["ESZ6"]                         # ESH7 did not trade: no bars
    assert settle.to_dict() == {("ESH7", day("2026-09-01")): 5000.0,   # days outside the window dropped
                                ("ESZ6", day("2026-09-01")): 5000.0}
