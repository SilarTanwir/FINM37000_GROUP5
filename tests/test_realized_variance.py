"""PLANNING.md §9: test_realized_variance (roadmap 10). Offline: simulated paths and bars only."""

import math

import numpy as np
import pandas as pd
import pytest

from esvrp.data.underlying import build_bars
from esvrp.vrp.realized import realized_variance, returns_from_bars, rv_table

TZ = "America/Chicago"
SYMBOL = "ESZ6"


def _ct(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=TZ).tz_convert("UTC")


@pytest.fixture
def bars5m(raw_week) -> pd.DataFrame:
    return build_bars(raw_week(), "5m", "full_globex")


@pytest.mark.parametrize("n_steps", [21, 21 * 78])  # daily and 5-minute-like sampling
def test_rv_converges_to_sigma_squared(gbm_paths, n_steps):
    sigma, tau = 0.18, 30 / 365
    paths = gbm_paths(sigma=sigma, tau=tau, n_steps=n_steps, n_paths=2000)
    rvs = [realized_variance(pd.Series(np.diff(np.log(p))), tau) for p in paths]
    assert np.mean(rvs) == pytest.approx(sigma**2, rel=0.03)


def test_log_and_simple_rv_agree_on_small_returns(gbm_paths):
    tau = 30 / 365
    p = gbm_paths(sigma=0.18, tau=tau, n_steps=21, n_paths=1)[0]
    log_rv = realized_variance(np.diff(np.log(p)), tau)
    simple_rv = realized_variance(p[1:] / p[:-1] - 1, tau)
    assert simple_rv == pytest.approx(log_rv, rel=0.02)


def test_realized_variance_basic_and_edge_cases():
    assert realized_variance([0.01, -0.02], tau=0.5) == pytest.approx((0.01**2 + 0.02**2) / 0.5)
    assert realized_variance(pd.Series([0.01, np.nan, -0.02]), 0.5) == pytest.approx(0.0005 / 0.5)
    assert math.isnan(realized_variance([], 0.5))
    assert math.isnan(realized_variance([0.01], 0.0))
    assert math.isnan(realized_variance([0.01], float("nan")))


def test_gap_handling(bars5m):
    """Halts, the weekend and the Sunday reopen each give one spanning return: none dropped or doubled."""
    r = returns_from_bars(bars5m, "5m", "full_globex", underlying=SYMBOL)
    assert len(r) == len(bars5m) - 1
    closes = bars5m.set_index("ts")["close"]
    # Telescoping: the returns sum to the log move from the first to the last bar, which fails if any
    # gap return is missing or counted twice.
    assert r.sum() == pytest.approx(math.log(closes.iloc[-1] / closes.iloc[0]), abs=1e-12)

    def spanning(after: str, before: str) -> None:
        a, b = _ct(after), _ct(before)
        assert b in closes.index and a in closes.index
        assert not ((closes.index > a) & (closes.index < b)).any()  # nothing traded in between
        assert r[b] == pytest.approx(math.log(closes[b] / closes[a]))

    spanning("2026-10-13 16:00", "2026-10-13 17:05")  # maintenance halt, Tuesday evening
    spanning("2026-10-09 16:00", "2026-10-11 17:05")  # Friday close to Sunday reopen


def test_window_starts_at_snapshot_and_stops_at_expiry(bars5m):
    snap, exp = _ct("2026-10-12 15:00"), _ct("2026-10-16 15:00")
    r = returns_from_bars(bars5m, "5m", start=snap, end=exp, underlying=SYMBOL)
    closes = bars5m.set_index("ts")["close"]
    in_window = closes.index[(closes.index > snap) & (closes.index <= exp)]
    assert list(r.index) == list(in_window)
    assert r.index.max() == exp  # nothing after the expiry (the 15:00-16:00 CT hour is cut)
    assert r.sum() == pytest.approx(math.log(closes[exp] / closes[snap]), abs=1e-12)
    assert closes.index.max() > exp  # the contract does trade after the option expires


def test_rv_table_matches_manual_window(bars5m):
    snap, exp = _ct("2026-10-12 15:00"), _ct("2026-10-16 15:00")
    tau = (exp - snap).total_seconds() / (365 * 86400)
    chains = pd.DataFrame({"date": [pd.Timestamp("2026-10-12").date()], "expiry": [exp],
                           "underlying": [SYMBOL], "snapshot_ts": [snap], "tau": [tau]})
    daily = pd.DataFrame({"underlying": [SYMBOL], "date": [pd.Timestamp("2026-10-12").date()],
                          "settle": [5000.0]})
    out = rv_table(chains, bars5m, daily)
    r = returns_from_bars(bars5m, "5m", start=snap, end=exp, underlying=SYMBOL)
    row = out.iloc[0]
    assert row["rv_5m"] == pytest.approx((r**2).sum() / tau)
    assert row["n_5m"] == len(r)
    assert row["start_gap_s"] == 0 and row["end_gap_s"] == 0
    assert row["rv_5m_simple"] == pytest.approx(row["rv_5m"], rel=0.02)
    assert math.isnan(row["rv_daily"]) and row["n_daily"] == 0  # a single settlement gives no return


def test_rv_table_flags_gaps_and_missing_contracts(bars5m):
    snap, exp = _ct("2026-10-12 15:00"), _ct("2026-10-16 15:00") + pd.Timedelta(minutes=2)
    chains = pd.DataFrame({
        "date": [pd.Timestamp("2026-10-12").date()] * 2, "expiry": [exp, exp + pd.Timedelta(days=1)],
        "underlying": [SYMBOL, "ESH7"], "snapshot_ts": [snap, snap], "tau": [4 / 365, 5 / 365],
    })
    out = rv_table(chains, bars5m, bars5m.iloc[0:0].assign(settle=[]))
    assert out.loc[0, "end_gap_s"] == 120  # expiry falls inside a bar: the last bar ended 2 min earlier
    assert out.loc[1, "n_5m"] == 0 and math.isnan(out.loc[1, "rv_5m"])  # no bars for this contract


def _daily(dates: list[str], settles: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"underlying": SYMBOL, "date": [pd.Timestamp(d).date() for d in dates],
                         "settle": settles})


def test_daily_returns_skip_days_without_settlement():
    daily = _daily(["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16"],
                   [5000.0, 5050.0, np.nan, 4990.0, 5010.0])  # a shortened holiday session: no settle
    r = returns_from_bars(daily, "1d", start="2026-10-12", end="2026-10-16", underlying=SYMBOL)
    assert len(r) == 3
    assert r.sum() == pytest.approx(math.log(5010.0 / 5000.0))
    assert r.iloc[1] == pytest.approx(math.log(4990.0 / 5050.0))  # one return spanning the missing day
    assert returns_from_bars(daily, "daily", start="2026-10-12", end="2026-10-16",
                             underlying=SYMBOL).equals(r)


def test_daily_window_uses_last_settlement_on_or_before_each_end():
    daily = _daily(["2026-10-09", "2026-10-13", "2026-10-16", "2026-10-19"], [100.0, 110.0, 121.0, 150.0])
    r = returns_from_bars(daily, "1d", start="2026-10-11", end="2026-10-17", underlying=SYMBOL)
    assert len(r) == 2  # base 10-09, then 10-13 and 10-16; 10-19 is after the end
    assert r.sum() == pytest.approx(math.log(121.0 / 100.0))


def test_simple_kind(bars5m):
    snap, exp = _ct("2026-10-12 15:00"), _ct("2026-10-16 15:00")
    log_r = returns_from_bars(bars5m, "5m", start=snap, end=exp, underlying=SYMBOL)
    simple = returns_from_bars(bars5m, "5m", start=snap, end=exp, underlying=SYMBOL, kind="simple")
    assert np.allclose(np.expm1(log_r.to_numpy()), simple.to_numpy())


def test_bad_arguments(bars5m):
    with pytest.raises(ValueError):
        returns_from_bars(bars5m, "1h")
    with pytest.raises(ValueError):
        returns_from_bars(bars5m, "5m", session="nope")
    with pytest.raises(ValueError):
        returns_from_bars(bars5m, "5m", kind="percent")
    assert returns_from_bars(bars5m.iloc[0:0], "5m").empty


# --- windows the bars do not cover (Tanwir's review of #36) ------------------------------------------

def _one_chain(expiry, snap=None, underlying=SYMBOL):
    snap = snap if snap is not None else _ct("2026-10-12 15:00")
    return pd.DataFrame({"date": [pd.Timestamp("2026-10-12").date()], "expiry": [expiry],
                         "underlying": [underlying], "snapshot_ts": [snap],
                         "tau": [(expiry - snap).total_seconds() / (365 * 86400)]})


def test_rv_is_empty_when_the_option_expires_after_the_bars_end(bars5m):
    """Summing the returns that exist and dividing by the full tau would understate RV; leave it empty."""
    last_bar = bars5m["ts"].max()
    out = rv_table(_one_chain(last_bar + pd.Timedelta(days=10)), bars5m, bars5m.iloc[0:0].assign(settle=[]))
    row = out.iloc[0]
    assert math.isnan(row["rv_5m"]) and math.isnan(row["rv_5m_simple"])
    assert not row["complete_5m"] and row["end_gap_s"] == pytest.approx(10 * 86400)
    assert row["n_5m"] > 0  # the returns that exist are still counted


def test_tolerance_is_configurable_and_a_weekend_is_not_a_gap(bars5m):
    last_bar = bars5m["ts"].max()
    chains = _one_chain(last_bar + pd.Timedelta(days=1.5))
    daily = bars5m.iloc[0:0].assign(settle=[])
    assert rv_table(chains, bars5m, daily)["complete_5m"].iloc[0]  # within the default 2 days
    assert rv_table(chains, bars5m, daily)["rv_5m"].notna().iloc[0]
    strict = rv_table(chains, bars5m, daily, max_gap_days=1.0)
    assert not strict["complete_5m"].iloc[0] and math.isnan(strict["rv_5m"].iloc[0])


def test_rv_is_empty_when_the_snapshot_is_long_after_the_base_bar(bars5m):
    last_bar = bars5m["ts"].max()
    snap = last_bar + pd.Timedelta(days=5)
    out = rv_table(_one_chain(snap + pd.Timedelta(days=3), snap), bars5m, bars5m.iloc[0:0].assign(settle=[]))
    assert math.isnan(out["rv_5m"].iloc[0]) and out["start_gap_s"].iloc[0] > 4 * 86400


def test_daily_rv_is_empty_when_settlements_stop_early(bars5m):
    dates = ["2026-10-12", "2026-10-13", "2026-10-14"]
    daily = pd.DataFrame({"underlying": SYMBOL, "date": [pd.Timestamp(d).date() for d in dates],
                          "settle": [5000.0, 5010.0, 5005.0]})
    expiry = _ct("2026-10-16 15:00")
    row = rv_table(_one_chain(expiry), bars5m, daily).iloc[0]
    assert row["complete_5m"] and row["rv_5m"] > 0  # the 5-minute bars cover the window
    assert row["complete_daily"] and row["rv_daily"] > 0  # last settlement 10-14 is 2 days before 10-16
    late = rv_table(_one_chain(_ct("2026-10-19 15:00")), bars5m, daily, max_gap_days=2).iloc[0]
    assert not late["complete_daily"] and math.isnan(late["rv_daily"])
    assert late["end_gap_days"] == pytest.approx(5.0) and late["n_daily"] == 2


def test_complete_and_incomplete_windows_side_by_side(bars5m):
    snap, last_bar = _ct("2026-10-12 15:00"), bars5m["ts"].max()
    chains = pd.concat([_one_chain(_ct("2026-10-16 15:00")), _one_chain(last_bar + pd.Timedelta(days=20))])
    out = rv_table(chains, bars5m, bars5m.iloc[0:0].assign(settle=[]))
    assert out["complete_5m"].tolist() == [True, False]
    assert out["rv_5m"].notna().tolist() == [True, False]
    assert snap < last_bar
