"""Tests for data/quotes.py (PLANNING.md §6 stage 2, roadmap 5). Fixtures only, no network."""

import pandas as pd
import pytest

from esvrp.data.quotes import clean_quotes, futures_mid_at_snapshot, load_snapshots, snapshot_ts

UNIVERSE = pd.DataFrame({
    "instrument_id": [1, 2, 3],
    "raw_symbol": ["EWV6 C5000", "EWV6 P5000", "EWV6 C5100"],
    "cp": ["c", "p", "c"],
    "strike": [5000.0, 5000.0, 5100.0],
    "expiry": pd.to_datetime(["2026-10-30 20:00"] * 3, utc=True),
    "underlying": ["ESZ6"] * 3,
    "user_defined": [False, False, False],
})


def bbo(rows):
    """bbo-1m records: (utc time, instrument_id, bid, ask)."""
    df = pd.DataFrame(rows, columns=["ts_recv", "instrument_id", "bid_px_00", "ask_px_00"])
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
    df["bid_sz_00"] = df["ask_sz_00"] = 1
    return df.set_index("ts_recv")


def test_snapshot_time_handles_daylight_saving():
    assert snapshot_ts("2026-09-01", "15:00") == pd.Timestamp("2026-09-01 20:00", tz="UTC")  # CDT
    assert snapshot_ts("2026-12-01", "15:00") == pd.Timestamp("2026-12-01 21:00", tz="UTC")  # CST


def test_last_quote_at_or_before_snapshot_with_backfill():
    q = load_snapshots(UNIVERSE, bbo([
        ("2026-09-01 19:58", 1, 10.0, 11.0),
        ("2026-09-01 20:00", 1, 12.0, 13.0),   # snapshot minute: wins
        ("2026-09-01 20:01", 1, 99.0, 99.5),   # after snapshot: ignored
        ("2026-09-01 19:50", 2, 20.0, 21.0),   # 10 min stale: back-filled
        ("2026-09-01 19:30", 3, 30.0, 31.0),   # 30 min stale: beyond 15-min look-back
    ]), "15:00", backfill_minutes=15)
    assert list(q["instrument_id"]) == [1, 2]
    row1 = q.set_index("instrument_id").loc[1]
    assert (row1["bid"], row1["ask"], row1["quote_age_s"]) == (12.0, 13.0, 0.0)
    assert q.set_index("instrument_id").loc[2, "quote_age_s"] == 600.0


def test_expired_options_excluded():
    uni = UNIVERSE.assign(expiry=pd.Timestamp("2026-09-01 20:00", tz="UTC"))
    assert load_snapshots(uni, bbo([("2026-09-01 20:00", 1, 1.0, 2.0)]), "15:00").empty


def test_futures_mid_at_snapshot():
    f = bbo([("2026-09-01 19:59", 9, 4999.0, 5001.0), ("2026-09-01 20:00", 9, 5000.0, 5000.5)])
    f["symbol"] = "ESZ6"
    out = futures_mid_at_snapshot(f.drop(columns="instrument_id"), "15:00")
    assert out.to_dict("records") == [{"date": pd.Timestamp("2026-09-01").date(),
                                       "underlying": "ESZ6", "F": 5000.25}]


def test_cleaning_rules_drop_bad_quotes():
    q = pd.DataFrame({
        "strike": [4900, 5000, 5100, 5200],
        "bid": [0.0, 10.0, 12.0, 5.0],   # zero bid
        "ask": [1.0, 11.0, 11.0, 6.0],   # crossed at 5100
    })
    cleaned, log = clean_quotes(q, cfg=None)
    assert list(cleaned["strike"]) == [5000, 5200]
    assert log["zero_bid"] == 1 and log["crossed"] == 1


def chain(F=5000.0):
    strikes = [4400.0 + 100 * i for i in range(13)]          # 4400..5600, 13 strikes
    return pd.DataFrame({
        "date": pd.Timestamp("2026-09-01").date(),
        "snapshot_ts": pd.Timestamp("2026-09-01 20:00", tz="UTC"),
        "expiry": pd.Timestamp("2026-10-30 20:00", tz="UTC"),
        "cp": ["p" if k < F else "c" for k in strikes],
        "strike": strikes,
        "bid": 5.0, "ask": 6.0, "F": F,
    })


def test_full_chain_survives_and_counts_are_logged():
    cleaned, log = clean_quotes(chain(), cfg=None)
    assert len(cleaned) == 13 and log["kept"] == 13
    assert set(log) >= {"no_quote", "zero_bid", "crossed", "locked", "no_futures_price", "below_intrinsic",
                        "chain_min_strikes", "chain_min_range", "kept", "one_sided_kept"}


def test_below_intrinsic_locked_and_one_sided():
    q = chain()
    q.loc[q["strike"] == 4400.0, ["cp", "bid", "ask"]] = ["c", 5.0, 6.0]   # deep ITM call priced at 5.5
    q.loc[q["strike"] == 5300.0, ["bid", "ask"]] = [5.0, 5.0]              # locked
    q.loc[q["strike"] == 5400.0, "ask"] = float("nan")                     # one-sided, kept
    cleaned, log = clean_quotes(q, cfg=None)
    assert log["below_intrinsic"] == 1 and log["locked"] == 1
    assert log["one_sided_kept"] == 1 and cleaned["one_sided"].sum() == 1


def test_thin_or_narrow_chains_dropped():
    _, log = clean_quotes(chain().iloc[:5], cfg=None)                    # 5 strikes < 10
    assert log["chain_min_strikes"] == 5 and log["kept"] == 0
    narrow = chain(F=5000.0)
    narrow = narrow[narrow["strike"].between(4600.0, 5600.0)]             # 11 strikes, low end too close to F
    _, log = clean_quotes(narrow, cfg=None)
    assert log["chain_min_range"] == len(narrow)


@pytest.mark.parametrize("flag,rule", [("drop_zero_bid", "zero_bid"), ("drop_crossed", "crossed")])
def test_rules_can_be_switched_off(flag, rule):
    from esvrp.data.quotes import _DefaultCleaning

    q = pd.DataFrame({"strike": [5000, 5100], "bid": [0.0, 12.0], "ask": [1.0, 11.0]})
    _, log = clean_quotes(q, _DefaultCleaning(**{flag: False}))
    assert log[rule] == 0


def test_missing_futures_price_counted_separately():
    far = chain().assign(expiry=pd.Timestamp("2027-12-31 21:00", tz="UTC"), F=float("nan"))
    q = pd.concat([chain(), far])
    cleaned, log = clean_quotes(q, cfg=None)
    assert log["no_futures_price"] == 13 and log["chain_min_range"] == 0 and len(cleaned) == 13


def test_failed_days_reported_after_other_days_finish(monkeypatch):
    from types import SimpleNamespace

    import esvrp.data.quotes as quotes

    done = []

    def fake_pull_day(cfg, universe, day, client):
        if str(day.date()) == "2026-09-02":
            raise RuntimeError("gateway timeout")
        done.append(str(day.date()))
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr(quotes, "_pull_day", fake_pull_day)
    cfg = SimpleNamespace(data=SimpleNamespace(start="2026-09-01", end="2026-09-04"))
    with pytest.raises(RuntimeError, match=r"1 day\(s\) failed.*2026-09-02"):
        quotes.pull_snapshots(cfg, UNIVERSE)
    assert sorted(done) == ["2026-09-01", "2026-09-03", "2026-09-04"]


def test_holiday_with_no_usable_quotes_returns_empty():
    # Labor Day 2026: one option record in the window, no prices, nothing at 15:00.
    holiday = bbo([("2026-09-07 19:50", 99, float("nan"), float("nan"))])
    assert load_snapshots(UNIVERSE, holiday, "15:00").empty
    holiday["symbol"] = "ESZ6"
    assert futures_mid_at_snapshot(holiday.drop(columns="instrument_id"), "15:00").empty


def test_completely_empty_pulls_return_empty():
    empty = bbo([]).assign(symbol=pd.Series(dtype=str))
    assert load_snapshots(UNIVERSE, empty, "15:00").empty
    assert futures_mid_at_snapshot(empty.drop(columns="instrument_id"), "15:00").empty
