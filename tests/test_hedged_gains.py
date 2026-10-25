"""Stage 6 driver: hedged gains over a chain (roadmap 12)."""

import numpy as np
import pandas as pd
import pytest

from esvrp.hedging.costs import apply_costs
from esvrp.hedging.simulator import _trailing_rv, build_hedged_gains, delta_hedged_gain
from esvrp.pricing.black76 import greeks, price

R, IV = 0.04, 0.18
EXPIRY = pd.Timestamp("2026-09-30 20:00", tz="UTC")  # 15:00 CT, last business day of September
LATE_EXPIRY = pd.Timestamp("2026-10-30 19:00", tz="UTC")  # after the sample's last snapshot
DAYS = pd.bdate_range("2026-09-01", "2026-09-30")
FUT = pd.Series(5000.0 + 10 * np.sin(np.arange(len(DAYS))), index=DAYS)


def make_chain(iv_path=None):
    """One futures contract, two expiries, strikes 4700-5300, a quote every business day."""
    rows = []
    for i, day in enumerate(DAYS):
        snap = pd.Timestamp(f"{day.date()} 20:00", tz="UTC")
        F = FUT.iloc[i]
        for expiry in (EXPIRY, LATE_EXPIRY):
            tau = (expiry - snap).total_seconds() / (365 * 86400)
            if tau <= 0:
                continue
            for k in np.arange(4700.0, 5301.0, 50.0):
                for cp in ("c", "p"):
                    iid = hash((expiry, k, cp)) % 10**8
                    iv = IV if iv_path is None else iv_path[i]
                    mid = price(F, k, tau, R, iv, cp)
                    g = greeks(F, k, tau, R, iv, cp)
                    rows.append({"instrument_id": iid, "raw_symbol": f"EW{k:.0f}{cp}", "underlying": "ESZ6",
                                 "date": day.date(), "snapshot_ts": snap, "expiry": expiry, "cp": cp,
                                 "strike": k, "moneyness": np.log(k / F), "tau": tau, "r": R, "F": F,
                                 "F_fut": F, "iv": iv, "delta": g["delta"], "vega": g["vega"],
                                 "bid": mid - 0.5, "ask": mid + 0.5, "mid": mid})
    return pd.DataFrame(rows)


def make_daily():
    """Settlements: July-August history for the trailing vol, then the September path."""
    aug = pd.bdate_range("2026-07-01", "2026-08-31")
    settle = pd.concat([pd.Series(5000.0 * np.exp(0.01 * np.cos(np.arange(len(aug)))), index=aug), FUT])
    return pd.DataFrame({"underlying": "ESZ6", "date": [d.date() for d in settle.index],
                         "settle": settle.to_numpy()})


def test_gain_matches_simulator_on_the_futures_path_and_expiry_settlement():
    chain, daily = make_chain(), make_daily()
    gains, _ = build_hedged_gains(chain, daily, hedge_deltas=("iv_fixed",))
    row = gains[(gains["date"] == DAYS[0].date()) & (gains["strike"] == 5000.0) & (gains["cp"] == "c")
                & (gains["expiry"] == EXPIRY)].iloc[0]

    # Rebalance at every snapshot before expiry (21 business days), pay off at the 30 Sep settlement.
    snaps = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} 20:00", tz="UTC") for d in DAYS[:-1]])
    path = pd.Series([*FUT.iloc[:-1], FUT.iloc[-1]], index=snaps.append(pd.DatetimeIndex([EXPIRY])))
    entry = chain[(chain["snapshot_ts"] == snaps[0]) & (chain["strike"] == 5000.0) & (chain["cp"] == "c")
                  & (chain["expiry"] == EXPIRY)].iloc[0]
    expected = delta_hedged_gain(entry, path, R, "iv_fixed", "mid")

    assert row["n_hedges"] == len(DAYS) - 1
    assert row["F_T"] == FUT.iloc[-1]
    assert row["gain"] == pytest.approx(expected.gain_dollars)
    assert row["gain_over_option"] == pytest.approx(expected.gain_over_option)
    assert row["cost"] == pytest.approx(-apply_costs(0.0, 0.5, len(DAYS) - 1, 0.5, 1.50))


def test_entries_are_filtered_and_drops_counted():
    chain = make_chain()
    chain.loc[(chain["date"] == DAYS[0].date()) & (chain["strike"] == 4950.0), "iv"] = np.nan
    chain.loc[(chain["date"] == DAYS[0].date()) & (chain["strike"] == 5050.0), "mid"] = np.nan
    gains, dropped = build_hedged_gains(chain, make_daily(), hedge_deltas=("iv_fixed",))

    days = gains["tau"] * 365
    assert days.between(14, 60).all() and (gains["moneyness"].abs() <= 0.10).all()
    assert (gains["expiry"] == EXPIRY).all()  # the October expiry is past the sample
    assert dropped["expires_after_sample"] > 0
    assert dropped["no_iv"] == 4 and dropped["one_sided"] == 4  # c and p, both expiries
    assert dropped["no_expiry_settle"] == 0


def test_missing_expiry_settlement_drops_the_option():
    daily = make_daily()
    daily = daily[daily["date"] != EXPIRY.tz_convert("America/Chicago").date()]
    gains, dropped = build_hedged_gains(make_chain(), daily, hedge_deltas=("iv_fixed",))
    assert gains.empty and dropped["no_expiry_settle"] > 0


def test_one_row_per_entry_and_hedge_delta():
    gains, _ = build_hedged_gains(make_chain(), make_daily())
    per_entry = gains.groupby(["instrument_id", "date"])["hedge_delta"].agg(sorted)
    assert (per_entry.map(tuple) == ("iv_daily", "iv_fixed", "rv")).all()


def test_iv_daily_follows_the_option_own_implied_vol():
    flat, _ = build_hedged_gains(make_chain(), make_daily(), hedge_deltas=("iv_daily", "iv_fixed"))
    g = flat.pivot_table(index=["instrument_id", "date"], columns="hedge_delta", values="gain")
    assert g["iv_daily"].to_numpy() == pytest.approx(g["iv_fixed"].to_numpy())

    rising = np.linspace(0.18, 0.30, len(DAYS))
    moved, _ = build_hedged_gains(make_chain(rising), make_daily(), hedge_deltas=("iv_daily", "iv_fixed"))
    g = moved.pivot_table(index=["instrument_id", "date"], columns="hedge_delta", values="gain")
    assert not np.allclose(g["iv_daily"], g["iv_fixed"])


def test_trailing_rv_uses_only_earlier_settlements():
    daily = make_daily()
    day = DAYS[10].date()
    before = _trailing_rv(daily)["ESZ6"]
    bumped = daily.copy()
    bumped.loc[bumped["date"] == day, "settle"] *= 1.05
    after = _trailing_rv(bumped)["ESZ6"]
    assert after[day] == pytest.approx(before[day])  # that day's own settlement is not used
    assert after[DAYS[11].date()] != pytest.approx(before[DAYS[11].date()])
    assert np.isfinite(before[DAYS[0].date()])  # July-August history covers the first September dates


def test_stage_run_writes_hedged_gains(tmp_path, monkeypatch):
    from esvrp import config
    from esvrp.hedging import simulator

    make_chain().to_parquet(tmp_path / "chain.parquet")
    make_daily().to_parquet(tmp_path / "underlying_daily.parquet")
    monkeypatch.setattr(simulator, "DATA_DIR", tmp_path)
    simulator.run(config.load_config())

    out = pd.read_parquet(tmp_path / "hedged_gains.parquet")
    assert not out.empty and set(out["hedge_delta"]) == {"iv_daily", "iv_fixed", "rv"}
    assert out["gain"].notna().all() and (out["cost"] > 0).all()
