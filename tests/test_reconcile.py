"""Reconciliation of VRP and hedged gains, H3 (roadmap 15)."""

import numpy as np
import pandas as pd
import pytest

from esvrp.analysis.reconcile import chain_hedged_variance, monthly_subsample, reconcile


def chains(rng, n_months=36, noise=0.002, link=1.0):
    """Daily chains on end-of-month expiries; HG = link * VRP + noise, plus a constant skew gap."""
    rows = []
    for m in pd.period_range("2023-10", periods=n_months, freq="M"):
        last_bday = pd.bdate_range(m.start_time, m.end_time)[-1]  # EW expiry: 15:00 CT, last business day
        expiry = (last_bday + pd.Timedelta(hours=15)).tz_localize("America/Chicago").tz_convert("UTC")
        level = rng.normal(-0.01, 0.01)  # one shock per month, shared by its overlapping windows
        start = m.to_timestamp(how="start")
        for d in pd.bdate_range(start - pd.Timedelta(days=30), start + pd.Timedelta(days=10)):
            vrp = level + rng.normal(0, 0.002)
            rows.append({"date": d.date(), "expiry": expiry, "vrp_diff": vrp, "sw": 0.04, "rv": 0.04 + vrp,
                         "hg": link * vrp + 0.004 + rng.normal(0, noise)})
    df = pd.DataFrame(rows).drop_duplicates(["date", "expiry"])
    return df[["date", "expiry", "vrp_diff", "sw", "rv"]], df[["date", "expiry", "hg"]]


def test_chain_hedged_variance_sums_before_dividing():
    exp = pd.Timestamp("2026-09-30 20:00", tz="UTC")
    g = pd.DataFrame({"date": [pd.Timestamp("2026-09-01").date()] * 3, "expiry": [exp] * 3,
                      "underlying": "ESZ6", "tau": 29 / 365, "moneyness": [0.0, -0.05, 0.12],
                      "hedge_delta": ["iv_fixed", "iv_fixed", "iv_daily"], "iv": [0.2, 0.3, 0.2],
                      "gain": [50.0, 150.0, 999.0], "gamma_dt": [1.0, 3.0, 1.0]})
    out = chain_hedged_variance(g).iloc[0]
    assert out["n_options"] == 2  # the iv_daily row is another hedge
    assert out["hg"] == pytest.approx((1.0 + 3.0) / (1.0 + 3.0))  # (50 + 150) / 50 points over weight 4
    assert out["iv2_gw"] == pytest.approx((1 * 0.04 + 3 * 0.09) / 4)
    assert out["rv_gw"] == pytest.approx(out["hg"] + out["iv2_gw"])
    narrow = chain_hedged_variance(g, moneyness_max=0.02).iloc[0]
    assert narrow["n_options"] == 1 and narrow["hg"] == pytest.approx(1.0)


def test_monthly_subsample_is_first_date_in_the_expiry_month():
    vrp, hg = chains(np.random.default_rng(0), n_months=3)
    m = monthly_subsample(vrp)
    assert len(m) == 3
    assert all(pd.Timestamp(d).day <= 3 for d in m["date"])  # first business day of each month
    expiry_months = pd.to_datetime(m["expiry"]).dt.tz_convert("America/Chicago").dt.month
    assert (pd.to_datetime(m["date"]).dt.month == expiry_months).all()


def test_linked_series_agree_and_correlate():
    vrp, hg = chains(np.random.default_rng(1))
    j, stats = reconcile(vrp, hg)
    daily, monthly = stats.loc["all chains (overlapping)"], stats.loc["monthly (non-overlapping)"]
    assert daily["n"] == len(j) and monthly["n"] == 36
    assert daily["pearson"] > 0.9 and daily["ci_low"] > 0.8
    assert daily["ci_low"] < daily["pearson"] < daily["ci_high"]
    assert monthly["ci_method"] == "Fisher z" and monthly["ci_low"] > 0.5
    assert daily["mean_hg"] - daily["mean_vrp"] == pytest.approx(0.004, abs=0.001)  # the skew gap
    assert abs(daily["t_vrp_hac"]) < abs(daily["t_vrp_naive"])  # overlap inflates the naive t


def test_unrelated_series_have_an_interval_covering_zero():
    vrp, hg = chains(np.random.default_rng(2), noise=0.02, link=0.0)
    _, stats = reconcile(vrp, hg)
    daily = stats.loc["all chains (overlapping)"]
    assert daily["ci_low"] < 0 < daily["ci_high"]


def test_bootstrap_interval_is_wider_than_iid_would_be():
    """Resampling blocks of dates keeps the shared monthly shocks; an iid interval would be too narrow."""
    vrp, hg = chains(np.random.default_rng(3), noise=0.01)
    j, stats = reconcile(vrp, hg)
    daily = stats.loc["all chains (overlapping)"]
    z = np.arctanh(daily["pearson"])
    iid_width = np.tanh(z + 1.96 / np.sqrt(len(j) - 3)) - np.tanh(z - 1.96 / np.sqrt(len(j) - 3))
    assert daily["ci_high"] - daily["ci_low"] > iid_width


def test_missing_and_unmatched_chains_are_dropped():
    vrp, hg = chains(np.random.default_rng(4), n_months=2)
    vrp.loc[vrp.index[:5], "vrp_diff"] = np.nan
    j, _ = reconcile(vrp, hg.iloc[10:])
    assert len(j) == len(vrp) - 10 and j["vrp"].notna().all()
