"""H2 bucket table (#16) on simulated hedged gains."""

import numpy as np
import pandas as pd
import pytest

from esvrp.analysis.buckets import BUCKET_COLUMNS, bucket_table, moneyness_bucket, sw_by_date

DATES = pd.bdate_range("2024-01-01", periods=120)
STRIKES = np.linspace(-0.08, 0.08, 9)  # moneyness grid


def make_gains(rng, effect=None, hedge_deltas=("iv_daily", "iv_fixed")):
    """Gains with a known per-vega mean by moneyness bucket; two maturities per date."""
    effect = effect or {"otm_put": -1.0, "near_atm": -0.5, "otm_call": 0.0}
    rows = []
    for d in DATES:
        for days in (21, 45):
            for m in STRIKES:
                bucket = moneyness_bucket(pd.Series([m])).iloc[0]
                vega = 5.0
                per_vega = effect[bucket] + 0.1 * rng.standard_normal()
                for hd in hedge_deltas:
                    rows.append({"date": d.date(), "hedge_delta": hd, "moneyness": m, "tau": days / 365,
                                 "vega": vega, "gain": per_vega * vega * 50 * 0.01,
                                 "gain_over_F": per_vega * 1e-4})
    return pd.DataFrame(rows)


def make_vrp(sw):
    """Two chains per date; the 30-day one carries `sw`, the 60-day one a decoy."""
    rows = []
    for d, s in zip(DATES, sw, strict=True):
        rows.append({"date": d.date(), "tau": 28 / 365, "sw": s})
        rows.append({"date": d.date(), "tau": 60 / 365, "sw": 99.0})
    return pd.DataFrame(rows)


def test_moneyness_bucket_edges():
    m = pd.Series([-0.03, -0.025, 0.0, 0.025, 0.026, np.nan])
    assert moneyness_bucket(m).tolist()[:5] == ["otm_put", "near_atm", "near_atm", "near_atm", "otm_call"]
    assert pd.isna(moneyness_bucket(m).iloc[5])


def test_sw_by_date_takes_the_chain_nearest_30_days():
    sw = sw_by_date(make_vrp(np.arange(len(DATES), dtype=float)))
    assert len(sw) == len(DATES)
    assert (sw.to_numpy() == np.arange(len(DATES))).all()


def test_bucket_means_recover_the_simulated_effect(rng):
    table = bucket_table(make_gains(rng), make_vrp(np.linspace(0.01, 0.05, len(DATES))))
    assert list(table.columns) == BUCKET_COLUMNS
    allr = table[(table["regime"] == "all") & (table["measure"] == "per_vega")].set_index(
        ["moneyness", "maturity"])
    assert len(allr) == 6
    for (mb, _), row in allr.iterrows():
        expected = {"otm_put": -1.0, "near_atm": -0.5, "otm_call": 0.0}[mb]
        assert row["mean"] == pytest.approx(expected, abs=0.02)
    # The baseline hedge only: one row per option and date in each cell.
    assert allr.loc[("near_atm", "14-30d"), "n"] == len(DATES) * 3
    assert allr.loc[("near_atm", "14-30d"), "n_dates"] == len(DATES)
    assert allr.loc[("otm_put", "14-30d"), "t_hac"] < -10
    assert allr.loc[("otm_put", "14-30d"), "hac_lags"] == 20
    assert allr.loc[("otm_put", "31-60d"), "hac_lags"] == 40
    gof = table[(table["regime"] == "all") & (table["measure"] == "gain_over_F")]
    assert gof["mean"].min() == pytest.approx(-1e-4, rel=0.05)


def test_regimes_split_the_sample_without_look_ahead(rng):
    gains = make_gains(rng)
    sw = np.linspace(0.01, 0.05, len(DATES))  # rising, so every date after warm-up is "high"
    table = bucket_table(gains, make_vrp(sw), min_periods=21)
    cell = table[(table["moneyness"] == "near_atm") & (table["maturity"] == "14-30d")
                 & (table["measure"] == "per_vega")].set_index("regime")
    assert cell.loc["high", "n_dates"] == len(DATES) - 20
    assert cell.loc["low", "n"] == 0 and cell.loc["mid", "n"] == 0

    # Appending later swap rates leaves earlier regimes, and so earlier-date cells, unchanged.
    early = gains[pd.to_datetime(gains["date"]) < DATES[80]]
    short = bucket_table(early, make_vrp(sw)[lambda v: pd.to_datetime(v["date"]) < DATES[80]])
    full = bucket_table(early, make_vrp(sw))
    pd.testing.assert_frame_equal(short, full)


def test_regime_follows_entry_date_swap_rate(rng):
    gains = make_gains(rng, effect={"otm_put": 0.0, "near_atm": 0.0, "otm_call": 0.0})
    # Random swap rates: regimes are mixed, and after the warm-up their counts add up to "all".
    sw = 0.02 + 0.01 * rng.random(len(DATES))
    table = bucket_table(gains, make_vrp(sw))
    cell = table[(table["moneyness"] == "otm_call") & (table["maturity"] == "31-60d")
                 & (table["measure"] == "per_vega")].set_index("regime")
    assert cell.loc[["low", "mid", "high"], "n"].sum() == cell.loc["all", "n"] - 20 * 3
    assert (cell.loc[["low", "mid", "high"], "n"] > 0).all()
