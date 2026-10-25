"""Roadmap 11: VRP table, constant maturity, descriptives and figures. Offline: simulated data only."""

import math

import numpy as np
import pandas as pd
import pytest

from esvrp.data.underlying import build_bars
from esvrp.reporting.figures import rv_comparison_figure, sw_vs_rv_figure, vrp_distribution_figure
from esvrp.vrp.premium import (
    constant_maturity,
    constant_maturity_rv,
    vrp_descriptives,
    vrp_table,
)
from esvrp.vrp.realized import returns_from_bars, rv_table

TZ = "America/Chicago"
DAY = pd.Timestamp("2026-10-12").date()


def _ct(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=TZ).tz_convert("UTC")


SNAP = _ct("2026-10-12 15:00")


def _strips(
    days_sw: list[tuple[float, float]], underlyings: list[str] | None = None, day=DAY
) -> pd.DataFrame:
    """One row per expiry for one date: (days to expiry, sw)."""
    names = underlyings or ["ESZ6"] * len(days_sw)
    rows = []
    for (days, sw), name in zip(days_sw, names, strict=True):
        rows.append({
            "date": day, "expiry": SNAP + pd.Timedelta(days=days), "underlying": name,
            "tau": days / 365, "snapshot_ts": SNAP, "sw_mid": sw, "sw_bid": sw * 0.9, "sw_ask": sw * 1.1,
        })
    return pd.DataFrame(rows)


# --- vrp_table -----------------------------------------------------------------------------------------

def _pair():
    exp = SNAP + pd.Timedelta(days=30)
    strips = pd.DataFrame({"date": [DAY, DAY], "expiry": [exp, exp + pd.Timedelta(days=30)],
                           "underlying": ["ESZ6"] * 2, "tau": [30 / 365, 60 / 365],
                           "sw_mid": [0.04, 0.05], "sw_ask": [0.05, 0.06], "n_strikes": [100, 90]})
    rv = pd.DataFrame({"date": [DAY, DAY], "expiry": [exp, exp + pd.Timedelta(days=30)],
                       "rv_5m": [0.03, 0.06], "rv_daily": [0.02, 0.07], "n_5m": [500, 900]})
    return strips, rv


def test_vrp_table_hand_computed():
    strips, rv = _pair()
    t = vrp_table(strips, rv)
    assert t["sw"].tolist() == [0.04, 0.05] and t["rv"].tolist() == [0.03, 0.06]
    assert t["vrp_diff"].to_numpy() == pytest.approx([-0.01, 0.01])
    assert t["vrp_log"].to_numpy() == pytest.approx([math.log(0.03 / 0.04), math.log(0.06 / 0.05)])
    assert t["vrp_swap_return"].to_numpy() == pytest.approx([0.03 / 0.04 - 1, 0.06 / 0.05 - 1])
    assert t["vrp"].equals(t["vrp_diff"])  # difference is the primary definition
    assert {"n_strikes", "n_5m", "underlying", "tau"} <= set(t.columns)


@pytest.mark.parametrize(
    "kind,col", [("diff", "vrp_diff"), ("log", "vrp_log"), ("swap_return", "vrp_swap_return")]
)
def test_vrp_kind_selects_column(kind, col):
    strips, rv = _pair()
    t = vrp_table(strips, rv, kind)
    assert t["vrp"].equals(t[col])


def test_vrp_variants_and_validation():
    strips, rv = _pair()
    t = vrp_table(strips, rv, sw_col="sw_ask", rv_col="rv_daily")
    assert t["vrp_diff"].to_numpy() == pytest.approx([0.02 - 0.05, 0.07 - 0.06])
    with pytest.raises(ValueError):
        vrp_table(strips, rv, "percent")
    with pytest.raises(ValueError):
        vrp_table(strips, rv, sw_col="sw_nope")
    with pytest.raises(ValueError):
        vrp_table(strips, rv.drop(columns="rv_5m"))


def test_vrp_negative_or_missing_inputs_give_nan_log():
    strips, rv = _pair()
    rv.loc[0, "rv_5m"] = 0.0
    strips.loc[1, "sw_mid"] = np.nan
    t = vrp_table(strips, rv)
    assert t["vrp_diff"].iloc[0] == pytest.approx(-0.04)
    assert t["vrp_log"].isna().all() and math.isnan(t["vrp_swap_return"].iloc[1])


def test_vrp_table_joins_only_matching_chains():
    strips, rv = _pair()
    t = vrp_table(strips, rv.iloc[:1])
    assert len(t) == 1


# --- constant maturity ----------------------------------------------------------------------------------

def test_constant_maturity_matches_eq_10_by_hand():
    cm = constant_maturity(_strips([(20, 0.04), (50, 0.05)]))
    w1, w2 = (50 - 30) / 30, (30 - 20) / 30
    expected = (0.04 * 20 / 365 * w1 + 0.05 * 50 / 365 * w2) / (30 / 365)
    assert len(cm) == 1 and not cm["extrapolated"].iloc[0]
    assert cm["sw_mid"].iloc[0] == pytest.approx(expected)
    assert cm["sw_ask"].iloc[0] == pytest.approx(expected * 1.1)  # linear, so variants scale through
    assert cm["tau"].iloc[0] == pytest.approx(30 / 365)
    assert cm["expiry"].iloc[0] == SNAP + pd.Timedelta(days=30)


@pytest.mark.parametrize("days", [[(20, 0.04), (50, 0.04)], [(5, 0.04), (35, 0.04), (66, 0.04)],
                                  [(35, 0.04), (66, 0.04)], [(10, 0.04), (20, 0.04)]])
def test_flat_variance_gives_the_same_variance_at_any_maturity(days):
    """Total variance is linear in time under a flat swap rate, so interpolation and extrapolation keep SW."""
    cm = constant_maturity(_strips(days))
    assert cm["sw_mid"].iloc[0] == pytest.approx(0.04)


def test_expiries_under_eight_days_are_dropped_and_the_next_two_extrapolate():
    strips = _strips([(5, 0.03), (35, 0.04), (66, 0.05)], ["ESZ6", "ESH7", "ESM7"])
    cm = constant_maturity(strips)
    assert cm["extrapolated"].iloc[0]
    assert cm["tau1"].iloc[0] == pytest.approx(35 / 365) and cm["tau2"].iloc[0] == pytest.approx(66 / 365)
    assert cm["underlying"].iloc[0] == "ESH7"  # underlying of the first valid expiry
    w1, w2 = (66 - 30) / 31, (30 - 35) / 31  # w2 is negative: extrapolation below both expiries
    expected = (0.04 * 35 * w1 + 0.05 * 66 * w2) / 30
    assert cm["sw_mid"].iloc[0] == pytest.approx(expected)


def test_bracketing_pair_is_preferred_over_the_nearest_two():
    cm = constant_maturity(_strips([(12, 0.03), (40, 0.04), (70, 0.05)]))
    assert not cm["extrapolated"].iloc[0]
    assert cm["tau1"].iloc[0] == pytest.approx(12 / 365) and cm["tau2"].iloc[0] == pytest.approx(40 / 365)


def test_every_expiry_before_the_target_extrapolates_from_the_last_two():
    cm = constant_maturity(_strips([(10, 0.03), (15, 0.035), (24, 0.04)]))
    assert cm["extrapolated"].iloc[0]
    assert cm["tau1"].iloc[0] == pytest.approx(15 / 365) and cm["tau2"].iloc[0] == pytest.approx(24 / 365)


def test_dates_without_two_valid_expiries_are_skipped_and_nan_sw_is_ignored():
    day2 = pd.Timestamp("2026-10-13").date()
    strips = pd.concat([
        _strips([(5, 0.03), (35, 0.04)]),
        _strips([(20, 0.04), (50, 0.05), (80, np.nan)], day=day2),
    ])
    cm = constant_maturity(strips)
    assert cm["date"].tolist() == [day2]  # the first date has only one expiry of at least 8 days
    assert cm["tau2"].iloc[0] == pytest.approx(50 / 365)


def test_constant_maturity_needs_a_sw_column():
    with pytest.raises(ValueError):
        constant_maturity(_strips([(20, 0.04), (50, 0.05)])[["date", "expiry", "tau", "underlying"]])


def test_constant_maturity_rv_uses_the_window_after_the_snapshot(raw_week):
    bars = build_bars(raw_week(), "5m", "full_globex")
    cm = constant_maturity(_strips([(2, 0.04), (5, 0.05)]), target_days=3, min_front_days=1)
    rv = constant_maturity_rv(cm, bars, bars.iloc[0:0].assign(settle=[]))
    window_end = SNAP + pd.Timedelta(days=3)  # Thursday 15:00 CT
    r = returns_from_bars(bars, "5m", start=SNAP, end=window_end, underlying="ESZ6")
    assert rv["rv_5m"].iloc[0] == pytest.approx((r**2).sum() / (3 / 365))
    assert rv["end_gap_s"].iloc[0] == 0
    t = vrp_table(cm, rv)  # joins on (date, expiry) like any other chain
    assert t["vrp_diff"].iloc[0] == pytest.approx(rv["rv_5m"].iloc[0] - cm["sw_mid"].iloc[0])


def test_constant_maturity_rv_of_nothing_is_empty():
    empty = constant_maturity(_strips([(5, 0.03)]))
    assert empty.empty
    assert constant_maturity_rv(empty, pd.DataFrame(columns=["underlying", "ts", "close"]),
                                pd.DataFrame(columns=["underlying", "date", "settle"])).empty


# --- descriptives ---------------------------------------------------------------------------------------

def test_descriptives_hand_computed():
    x = pd.DataFrame({"vrp_diff": [-0.02, -0.01, 0.0, 0.01, -0.04], "vrp_log": [np.nan] * 5,
                      "vrp_swap_return": [-0.5, -0.2, 0.1, 0.3, -0.6]})
    d = vrp_descriptives(x)
    row = d.loc["vrp_diff"]
    assert row["n"] == 5 and row["mean"] == pytest.approx(-0.012)
    assert row["std"] == pytest.approx(x["vrp_diff"].std())
    assert row["share_negative"] == pytest.approx(0.6)
    assert row["naive_t"] == pytest.approx(-0.012 / (x["vrp_diff"].std() / math.sqrt(5)))
    assert d.loc["vrp_log", "n"] == 0 and math.isnan(d.loc["vrp_log", "mean"])


# --- figures --------------------------------------------------------------------------------------------

def _chains(n=40, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=n)
    sw = 0.04 + 0.005 * rng.standard_normal(n)
    rv = sw - 0.005 + 0.01 * rng.standard_normal(n)
    df = pd.DataFrame({"date": dates.date, "sw": sw, "rv": rv, "vrp_diff": rv - sw,
                       "rv_5m": rv, "rv_daily": rv * (1 + 0.1 * rng.standard_normal(n))})
    return df


@pytest.mark.parametrize("make", [sw_vs_rv_figure, vrp_distribution_figure, rv_comparison_figure])
def test_figures_are_written_even_when_empty(make, tmp_path):
    out = make(_chains(), tmp_path / "figs" / "a.png")
    assert out.exists() and out.stat().st_size > 5000
    empty = make(_chains().iloc[:0], tmp_path / "empty.png")
    assert empty.exists()


def test_strip_table_carries_the_snapshot_time(flat_vol_chain):
    from esvrp.vrp.strip import strip_table

    chain, F, r, tau = flat_vol_chain(sigma=0.2)
    chain = chain.assign(date="2026-09-15", expiry="2026-10-30", underlying="ESZ6", F=F, r=r, tau=tau,
                         snapshot_ts=SNAP, bid=chain["mid"], ask=chain["mid"])
    assert strip_table(chain)["snapshot_ts"].iloc[0] == SNAP
    assert pd.isna(strip_table(chain.drop(columns="snapshot_ts"))["snapshot_ts"].iloc[0])


def test_rv_table_still_joins_to_strips(raw_week):
    """The keys of strip_table and rv_table line up so vrp_table can join them."""
    bars = build_bars(raw_week(), "5m", "full_globex")
    exp = SNAP + pd.Timedelta(days=3)
    chains = pd.DataFrame({"date": [DAY], "expiry": [exp], "underlying": ["ESZ6"], "snapshot_ts": [SNAP],
                           "tau": [3 / 365]})
    rv = rv_table(chains, bars, bars.iloc[0:0].assign(settle=[]))
    strips = chains.assign(sw_mid=0.04)
    assert len(vrp_table(strips, rv)) == 1
