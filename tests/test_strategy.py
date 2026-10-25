"""Strategy backtest and risk metrics (roadmap 16)."""

import numpy as np
import pandas as pd
import pytest

from esvrp.analysis.strategy import (
    margin_sensitivity,
    monthly_straddles,
    monthly_trades,
    monthly_trades_log,
    risk_metrics,
    run_strategy,
    run_structures,
    worst_months,
)

SEP = pd.Timestamp("2026-09-30 20:00", tz="UTC")  # 15:00 CT
OCT = pd.Timestamp("2026-10-30 19:00", tz="UTC")


def leg(date, expiry, strike, cp, gain, F=5000.0, vega=300.0, cost=40.0, iv=0.18, hd="iv_daily"):
    return {"date": pd.Timestamp(date).date(), "expiry": expiry, "underlying": "ESZ6", "strike": strike,
            "cp": cp, "F": F, "moneyness": np.log(strike / F), "tau": (expiry - pd.Timestamp(date, tz="UTC"))
            .total_seconds() / (365 * 86400), "iv": iv, "vega": vega, "gain": gain, "cost": cost,
            "hedge_delta": hd}


def gains():
    rows = []
    for d in ("2026-08-31", "2026-09-01", "2026-09-02"):  # 31 Aug is in the month before the expiry
        for k in (4950.0, 5000.0, 5050.0):
            for cp in ("c", "p"):
                rows.append(leg(d, SEP, k, cp, gain=100.0 if k == 5000.0 else 999.0))
    for d in ("2026-10-01", "2026-10-02"):
        for k in (5100.0, 5150.0):
            for cp in ("c", "p"):
                rows.append(leg(d, OCT, k, cp, gain=-250.0 if k == 5150.0 else 999.0, F=5140.0))
    rows.append(leg("2026-09-01", SEP, 5000.0, "c", gain=7.0, hd="rv"))  # another hedge delta
    return pd.DataFrame(rows)


def test_one_trade_per_month_on_the_first_snapshot_at_the_money():
    t = monthly_straddles(gains())
    assert list(t["date"]) == [pd.Timestamp("2026-09-01").date(), pd.Timestamp("2026-10-01").date()]
    assert list(t["strike"]) == [5000.0, 5150.0]
    assert (t["expiry"].iloc[0], t["expiry"].iloc[1]) == (SEP, OCT)


def test_short_pnl_costs_and_normalizations():
    t = monthly_straddles(gains(), margin_fraction=0.10).iloc[0]
    assert t["gross"] == pytest.approx(-200.0)  # seller earns minus the two long gains
    assert t["cost"] == pytest.approx(80.0) and t["net"] == pytest.approx(-280.0)
    dollar_vega = 600.0 * 50 * 0.01  # per vol point
    assert t["vega"] == pytest.approx(dollar_vega)
    assert t["net_per_vega"] == pytest.approx(-280.0 / dollar_vega)
    assert t["margin"] == pytest.approx(0.10 * 2 * 5000.0 * 50)
    assert t["net_return"] == pytest.approx(-280.0 / t["margin"])

    second = monthly_straddles(gains()).iloc[1]
    assert second["gross"] == pytest.approx(500.0) and second["net"] == pytest.approx(420.0)


def test_cost_multiplier_scales_costs_only():
    base, double = monthly_straddles(gains()), monthly_straddles(gains(), cost_multiplier=2.0)
    assert double["gross"].to_numpy() == pytest.approx(base["gross"].to_numpy())
    assert double["cost"].to_numpy() == pytest.approx(2 * base["cost"].to_numpy())


def test_strike_needs_both_legs():
    g = gains()
    g = g[~((g["strike"] == 5000.0) & (g["cp"] == "p"))]  # no put at the money in September
    assert monthly_straddles(g).iloc[0]["strike"] in (4950.0, 5050.0)


def test_no_trades_for_a_missing_hedge_delta():
    assert monthly_straddles(gains(), hedge_delta="iv_fixed").empty


def test_risk_metrics_by_hand():
    x = pd.Series([1.0, -3.0, 2.0, 2.0, -1.0, 3.0])
    m = risk_metrics(x, cvar_level=0.2)
    assert m["mean"] == pytest.approx(4 / 6)
    assert m["sharpe"] == pytest.approx(x.mean() / x.std() * np.sqrt(12))
    assert m["worst"] == -3.0 and m["second_worst"] == -1.0 and m["best"] == 3.0
    assert m["max_drawdown"] == pytest.approx(-3.0)  # from the peak of 1 down to -2
    assert m["cvar_20%"] == pytest.approx(-3.0)  # floor(0.2 * 6) = 1 worst month
    assert m["share_positive"] == pytest.approx(4 / 6)
    assert m["skew"] == pytest.approx(x.skew())


def test_drawdown_counts_losses_from_the_start():
    assert risk_metrics(pd.Series([-2.0, 1.0, -4.0]))["max_drawdown"] == pytest.approx(-5.0)


def test_run_strategy_reports_gross_and_net_series():
    res = run_strategy(gains())
    assert len(res.trades) == 2
    series = {"gross_per_vega", "net_per_vega", "gross", "net", "gross_return", "net_return"}
    assert series == set(res.metrics.index)
    assert res.metrics.loc["net", "mean"] == pytest.approx((-280.0 + 420.0) / 2)
    assert res.metrics.loc["net_per_vega", "units"] == "vol points"


# --- strangle as a robustness check, and the worst months ---------------------------------------------------

def leg2(date, expiry, strike, cp, gain, F=5000.0, vega=150.0, cost=30.0, iv=0.18, F_T=5100.0, hd="iv_daily"):
    row = leg(date, expiry, strike, cp, gain, F=F, vega=vega, cost=cost, iv=iv, hd=hd)
    return {**row, "F_T": F_T}


def chain_legs(expiry=SEP, date="2026-09-01", F=5000.0, F_T=5100.0, gains_by_strike=None, strikes=None):
    """Calls and puts at several strikes; ATM legs have more vega than the wings."""
    strikes = strikes or (4500.0, 4750.0, 4900.0, 5000.0, 5100.0, 5250.0, 5500.0)
    rows = []
    for k in strikes:
        for cp in ("c", "p"):
            gain = (gains_by_strike or {}).get((k, cp), 10.0)
            rows.append(leg2(date, expiry, k, cp, gain, F=F, F_T=F_T, vega=300.0 if k == F else 120.0))
    return pd.DataFrame(rows)


def test_strangle_sells_the_otm_call_and_put_nearest_five_percent():
    g = chain_legs(gains_by_strike={(5250.0, "c"): -40.0, (4750.0, "p"): 15.0})
    t = monthly_trades(g, "strangle")
    assert len(t) == 1
    r = t.iloc[0]
    assert (r["strike_call"], r["strike_put"]) == (5250.0, 4750.0)  # ln(1.05) and ln(0.95) are the targets
    assert r["structure"] == "strangle" and np.isnan(r["strike"]) and np.isnan(r["moneyness"])
    assert r["moneyness_call"] == pytest.approx(np.log(5250 / 5000)) and r["moneyness_put"] < 0
    assert r["gross"] == pytest.approx(-(-40.0 + 15.0))  # the seller earns minus the long gains
    assert r["cost"] == pytest.approx(60.0) and r["net"] == pytest.approx(25.0 - 60.0)
    assert r["vega"] == pytest.approx(240.0 * 50 * 0.01)
    assert r["net_per_vega"] == pytest.approx(r["net"] / r["vega"])
    assert r["margin"] == pytest.approx(0.10 * 2 * 5000.0 * 50)  # same assumed margin base as the straddle
    assert r["F_T"] == 5100.0 and r["move"] == pytest.approx(0.02)


def test_otm_distance_changes_the_strikes_and_the_atm_leg_is_never_used():
    g = chain_legs()
    near = monthly_trades(g, "strangle", otm=0.02).iloc[0]
    far = monthly_trades(g, "strangle", otm=0.10).iloc[0]
    assert (near["strike_call"], near["strike_put"]) == (5100.0, 4900.0)
    assert (far["strike_call"], far["strike_put"]) == (5500.0, 4500.0)
    assert near["strike_call"] > 5000.0 > near["strike_put"]  # strictly out of the money


def test_strangle_needs_legs_close_to_the_target_and_on_both_sides():
    sparse = chain_legs(strikes=(4950.0, 5000.0, 5050.0, 5500.0))  # nothing near +/-5% except 5500 (+9.5%)
    assert monthly_trades(sparse, "strangle").empty
    loose = monthly_trades(sparse, "strangle", tolerance=0.06)  # a looser tolerance accepts the nearest legs
    assert len(loose) == 1 and (loose.iloc[0]["strike_call"], loose.iloc[0]["strike_put"]) == (5050.0, 4950.0)
    only_calls = chain_legs()
    only_calls = only_calls[~((only_calls["cp"] == "p") & (only_calls["strike"] < 5000.0))]
    assert monthly_trades(only_calls, "strangle").empty


def test_straddle_columns_in_the_general_function():
    r = monthly_trades(chain_legs(), "straddle").iloc[0]
    assert r["structure"] == "straddle" and r["strike"] == r["strike_call"] == r["strike_put"] == 5000.0
    assert r["moneyness"] == pytest.approx(0.0) and r["vega"] == pytest.approx(600.0 * 50 * 0.01)
    assert monthly_trades(chain_legs()).equals(monthly_straddles(chain_legs()))


def test_unknown_structure_is_rejected_and_empty_input_gives_empty_trades():
    with pytest.raises(ValueError):
        monthly_trades(chain_legs(), "butterfly")
    assert monthly_trades(pd.DataFrame(columns=["hedge_delta"]), "strangle").empty


def test_run_structures_puts_the_straddle_next_to_strangles():
    lossy = {(5000.0, "c"): -100.0, (5000.0, "p"): -100.0}
    g = pd.concat([chain_legs(SEP, "2026-09-01", gains_by_strike=lossy),
                   chain_legs(OCT, "2026-10-01", F_T=4800.0)], ignore_index=True)
    s = run_structures(g, otms=(0.05, 0.02))
    assert list(s.index) == ["straddle (ATM)", "strangle (5% OTM)", "strangle (2% OTM)"]
    assert (s["n"] == 2).all()
    base = run_strategy(g)
    assert s.loc["straddle (ATM)", "net_mean"] == pytest.approx(base.metrics.loc["net_per_vega", "mean"])
    # The wings carry less vega, so the strangle is a smaller position in vega terms.
    assert s.loc["strangle (5% OTM)", "mean_vega_collected"] < s.loc["straddle (ATM)", "mean_vega_collected"]
    assert {"net_t_hac", "net_sharpe", "net_skew", "net_worst", "net_cvar"} <= set(s.columns)


def test_run_strategy_can_run_the_strangle_and_returns_the_worst_months():
    g = pd.concat([chain_legs(SEP, "2026-09-01", gains_by_strike={(5250.0, "c"): 500.0}),
                   chain_legs(OCT, "2026-10-01", F_T=4600.0)], ignore_index=True)
    res = run_strategy(g, structure="strangle", n_worst=1)
    assert len(res.trades) == 2 and (res.trades["structure"] == "strangle").all()
    assert len(res.worst) == 1 and res.worst.iloc[0]["rank"] == 1
    assert res.worst.iloc[0]["net_per_vega"] == res.trades["net_per_vega"].min()


def test_worst_months_lists_the_losers_with_the_move_and_the_implied_vol():
    trades = pd.DataFrame({
        "date": pd.to_datetime(["2026-03-02", "2026-04-01", "2026-05-01", "2026-06-01"]).date,
        "expiry": pd.to_datetime(["2026-03-31", "2026-04-30", "2026-05-29", "2026-06-30"]),
        "structure": "straddle", "net_per_vega": [1.0, -9.0, -2.0, 0.5], "net": [10.0, -90.0, -20.0, 5.0],
        "gross": [20.0, -70.0, -5.0, 15.0], "cost": [10.0, 20.0, 15.0, 10.0],
        "move": [0.01, -0.08, 0.03, 0.0], "iv_call": [0.14, 0.15, 0.16, 0.13],
        "iv_put": [0.16, 0.17, 0.16, 0.15],
        "days": [29.0, 29.0, 28.0, 29.0],
    })
    w = worst_months(trades, n=2)
    assert list(w["rank"]) == [1, 2] and list(w["net_per_vega"]) == [-9.0, -2.0]
    assert list(w["move"]) == [-0.08, 0.03] and w["iv_entry"].tolist() == pytest.approx([0.16, 0.16])
    assert list(w.columns) == ["rank", "date", "expiry", "structure", "net_per_vega", "net", "gross", "cost",
                               "move", "iv_entry", "days"]
    assert list(worst_months(trades, n=1, col="net")["net"]) == [-90.0]
    assert len(worst_months(trades, n=10)) == 4  # fewer trades than asked for
    empty = worst_months(trades.iloc[:0])
    assert empty.empty and list(empty.columns) == list(w.columns)
    assert worst_months(trades, col="not_a_column").empty


# --- margin sensitivity ----------------------------------------------------------------------------------

def _trades_with_pnl(net, F=5000.0):
    return pd.DataFrame({"F": F, "net": net, "gross": net})


def test_margin_table_reports_losses_as_a_share_of_margin_and_nothing_redundant():
    trades = _trades_with_pnl([800.0, 600.0, -1500.0, 400.0, 700.0, -3000.0, 500.0, 900.0])
    s = margin_sensitivity(trades, fractions=(0.06, 0.10))
    lo, hi = s.loc["6%"], s.loc["10%"]
    assert lo["margin_usd"] == pytest.approx(0.06 * 2 * 5000.0 * 50)
    for loss in ("worst", "cvar", "max_drawdown"):
        assert lo[f"{loss}_usd"] == pytest.approx(hi[f"{loss}_usd"])  # dollars do not depend on the margin
        assert lo[f"{loss}_pct_margin"] / hi[f"{loss}_pct_margin"] == pytest.approx(0.10 / 0.06)
        assert lo[f"{loss}_pct_margin"] == pytest.approx(lo[f"{loss}_usd"] / lo["margin_usd"])
    redundant = {"mean_monthly", "annualized", "t_hac", "sharpe", "skew"}
    assert not redundant & set(s.columns)  # one fraction only rescales these, so they are not reported


def test_margin_table_matches_the_trade_table_at_its_own_fraction():
    g = pd.concat([chain_legs(SEP, "2026-09-01"), chain_legs(OCT, "2026-10-01", F=5140.0)], ignore_index=True)
    res = run_strategy(g, margin_fraction=0.10)
    s = margin_sensitivity(res.trades, fractions=(0.10,))
    assert s.loc["10%", "worst_pct_margin"] == pytest.approx(res.metrics.loc["net_return", "worst"])
    drawdown = res.metrics.loc["net_return", "max_drawdown"]
    assert s.loc["10%", "max_drawdown_pct_margin"] == pytest.approx(drawdown)


def test_margin_table_shows_how_much_of_the_margin_a_drawdown_uses():
    trades = _trades_with_pnl([-12000.0, 3000.0, -9000.0])  # margin at 6% is $30,000
    s = margin_sensitivity(trades, fractions=(0.06, 0.10))
    assert s.loc["6%", "max_drawdown_usd"] == pytest.approx(-18000.0)  # peak 0 down to -18,000
    assert s.loc["6%", "max_drawdown_pct_margin"] == pytest.approx(-18000.0 / 30000.0)
    assert s.loc["10%", "max_drawdown_pct_margin"] == pytest.approx(-18000.0 / 50000.0)
    assert s.loc["6%", "worst_pct_margin"] == pytest.approx(-12000.0 / 30000.0)


def test_margin_table_with_no_trades():
    s = margin_sensitivity(pd.DataFrame(columns=["F", "net"]))
    assert list(s.index) == ["6%", "8%", "10%"] and (s["n"] == 0).all()


# --- units of the strangle tolerance, and the cost columns of the structure table ----------------------

def test_tolerance_is_in_log_moneyness_units():
    """0.02 in ln(K/F): about 2% of F. The 5% call target is ln(1.05) = 0.0488."""
    # ln(5150/5000) = 0.0296 and ln(5200/5000) = 0.0392, against a target of ln(1.05) = 0.0488
    g = chain_legs(strikes=(4750.0, 4900.0, 5000.0, 5150.0, 5200.0))
    near = monthly_trades(g, "strangle", otm=0.05, tolerance=0.02)  # |0.0296 - 0.0488| = 0.0192 <= 0.02
    assert len(near) == 1 and near.iloc[0]["strike_call"] == 5200.0  # nearer to the target than 5150
    assert monthly_trades(g, "strangle", otm=0.05, tolerance=0.005).empty  # 0.0392 is 0.0096 away: too far
    assert len(monthly_trades(g, "strangle", otm=0.05, tolerance=0.0096 + 1e-9)) == 1


def test_run_structures_shows_how_much_of_the_result_is_costs():
    lossless = {(k, cp): 0.0 for k in (4750.0, 5000.0, 5250.0) for cp in ("c", "p")}
    g = chain_legs(SEP, "2026-09-01", gains_by_strike=lossless)  # zero gains, so the result is only costs
    s = run_structures(g, otms=(0.05,))
    straddle, strangle = s.loc["straddle (ATM)"], s.loc["strangle (5% OTM)"]
    assert straddle["cost_usd"] == pytest.approx(60.0) and strangle["cost_usd"] == pytest.approx(60.0)
    assert straddle["cost_per_vega"] == pytest.approx(60.0 / (600.0 * 50 * 0.01))
    assert strangle["cost_per_vega"] == pytest.approx(60.0 / (240.0 * 50 * 0.01))
    assert strangle["cost_per_vega"] > straddle["cost_per_vega"]  # the same fixed costs on less vega
    assert straddle["cost_per_vega"] == pytest.approx(straddle["gross_mean"] - straddle["net_mean"])


# --- entry fallback: a month whose first snapshot has no straddle (#56 b) ---------------------------------

def _month(days_with_legs: dict, expiry=SEP, F=5000.0):
    """Legs by snapshot date. Each value is a list of (strike, cp) with gain 10 for each leg."""
    rows = []
    for date, legs in days_with_legs.items():
        for strike, cp in legs:
            rows.append(leg2(date, expiry, strike, cp, 10.0, F=F, vega=300.0 if strike == F else 120.0))
    return pd.DataFrame(rows)


BOTH = [(5000.0, "c"), (5000.0, "p")]
ONE_SIDED = [(5000.0, "c"), (5050.0, "p")]  # no strike with both a call and a put


def test_first_snapshot_is_used_when_it_has_a_straddle():
    t = monthly_trades(_month({"2026-09-01": BOTH, "2026-09-02": BOTH}))
    assert len(t) == 1 and t.iloc[0]["date"] == pd.Timestamp("2026-09-01").date()
    assert t.iloc[0]["entry_delay_days"] == 0


def test_next_snapshot_is_used_when_the_first_has_no_straddle():
    g = _month({"2026-09-01": ONE_SIDED, "2026-09-02": ONE_SIDED, "2026-09-03": BOTH})
    t = monthly_trades(g)
    assert len(t) == 1 and t.iloc[0]["date"] == pd.Timestamp("2026-09-03").date()
    assert t.iloc[0]["entry_delay_days"] == 2  # business days after the first snapshot
    assert monthly_trades(g, max_lookahead_days=1).empty  # the straddle is two snapshots away


def test_lookahead_is_in_business_days_after_the_first_snapshot():
    # First snapshot Fri 2026-09-04; the straddle appears Fri 2026-09-11 (5 business days later) or Mon 09-14.
    first = {"2026-09-04": ONE_SIDED}
    five = monthly_trades(_month({**first, "2026-09-11": BOTH}))
    assert len(five) == 1 and five.iloc[0]["entry_delay_days"] == 5
    six = monthly_trades(_month({**first, "2026-09-14": BOTH}))
    assert six.empty  # 6 business days after the first snapshot is beyond the default limit
    assert len(monthly_trades(_month({**first, "2026-09-14": BOTH}), max_lookahead_days=6)) == 1


def test_months_without_any_straddle_are_logged_with_a_reason():
    g = pd.concat([_month({"2026-09-01": ONE_SIDED, "2026-09-02": ONE_SIDED}, SEP),
                   _month({"2026-10-01": BOTH}, OCT, F=5140.0)], ignore_index=True)
    trades, skipped = monthly_trades_log(g)
    assert list(trades["expiry"]) == [OCT]
    assert len(skipped) == 1 and skipped.iloc[0]["expiry"] == SEP
    assert skipped.iloc[0]["reason"] == "no_strike_with_both_legs" and skipped.iloc[0]["snapshots_tried"] == 2
    assert skipped.iloc[0]["first_snapshot"] == pd.Timestamp("2026-09-01").date()


def test_expected_expiries_without_snapshots_are_logged_too():
    g = _month({"2026-09-01": BOTH})
    nov = pd.Timestamp("2026-11-30 20:00", tz="UTC")
    trades, skipped = monthly_trades_log(g, expected_expiries=[SEP, OCT, nov])
    assert len(trades) == 1
    assert set(skipped["expiry"]) == {OCT, nov} and (skipped["reason"] == "no_entries_in_month").all()
    # every expected month has either a trade or a logged reason
    assert set(trades["expiry"]) | set(skipped["expiry"]) == {SEP, OCT, nov}


def test_the_strangle_falls_back_too_and_logs_its_own_reason():
    wings = [(4750.0, "p"), (5250.0, "c")]
    no_wings = [(5000.0, "c"), (5000.0, "p")]
    t = monthly_trades(_month({"2026-09-01": no_wings, "2026-09-02": wings}), "strangle")
    assert len(t) == 1 and t.iloc[0]["entry_delay_days"] == 1
    _, skipped = monthly_trades_log(_month({"2026-09-01": no_wings}), "strangle")
    assert skipped.iloc[0]["reason"] == "no_otm_legs_within_tolerance"


def test_run_strategy_carries_the_log_and_the_delay_into_the_tables():
    g = pd.concat([_month({"2026-09-01": ONE_SIDED, "2026-09-02": BOTH}, SEP),
                   _month({"2026-10-01": ONE_SIDED}, OCT, F=5140.0)], ignore_index=True)
    res = run_strategy(g, expected_expiries=[SEP, OCT])
    assert len(res.trades) == 1 and res.trades.iloc[0]["entry_delay_days"] == 1
    assert list(res.skipped["expiry"]) == [OCT]
    s = run_structures(g)
    assert s.loc["straddle (ATM)", "n"] == 1 and s.loc["straddle (ATM)", "n_delayed"] == 1
    assert "entry_delay_days" in res.trades.columns


def test_a_later_entry_still_uses_that_days_legs():
    """The trade is priced on the snapshot it is entered on, not on the first one."""
    g = pd.concat([_month({"2026-09-01": ONE_SIDED}, SEP), _month({"2026-09-02": BOTH}, SEP, F=5010.0)],
                  ignore_index=True)
    t = monthly_trades(g).iloc[0]
    assert t["date"] == pd.Timestamp("2026-09-02").date() and t["F"] == 5010.0
