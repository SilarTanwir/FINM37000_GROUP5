"""Strategy backtest and risk metrics (roadmap 16)."""

import numpy as np
import pandas as pd
import pytest

from esvrp.analysis.strategy import monthly_straddles, risk_metrics, run_strategy

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
