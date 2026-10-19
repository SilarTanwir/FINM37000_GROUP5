"""PLANNING.md §9: test_chain_builder.

Definition parsing is tested in test_universe.py (roadmap 4) and cleaning rules in test_quotes.py
(roadmap 5). Chain assembly tests (parity-implied forward, IV, delta; roadmap 8) are here.
"""

import math

import numpy as np
import pandas as pd
import pytest

from esvrp.data.rates import risk_free_rate
from esvrp.pricing.black76 import greeks, price
from esvrp.pricing.chain import build_chain, forward_gap_table
from esvrp.pricing.forward import parity_forward

F, R, SIGMA = 5003.0, 0.04, 0.18
DAY = pd.Timestamp("2026-09-01").date()
SNAP = pd.Timestamp("2026-09-01 20:00", tz="UTC")
EXPIRY = pd.Timestamp("2026-09-30 20:00", tz="UTC")
TAU = 29 / 365


def quotes(F=F, sigma=SIGMA, expiry=EXPIRY, half_spread=0.25, F_fut=5000.0):
    """Cleaned stage 2 quotes for one chain, priced with Black-76 at a flat vol around forward F."""
    tau = (expiry - SNAP).total_seconds() / (365 * 86400)
    rows = []
    for strike in np.arange(4700.0, 5301.0, 50.0):
        for cp in ("c", "p"):
            mid = price(F, strike, tau, R, sigma, cp)
            rows.append({"date": DAY, "snapshot_ts": SNAP, "expiry": expiry, "underlying": "ESZ6", "cp": cp,
                         "strike": strike, "bid": mid - half_spread, "ask": mid + half_spread,
                         "F": F_fut, "one_sided": False})
    return pd.DataFrame(rows)


def test_parity_forward_recovers_the_forward():
    c = quotes().assign(mid=lambda q: (q["bid"] + q["ask"]) / 2)
    assert parity_forward(c, R, TAU) == pytest.approx(F, abs=1e-9)
    assert parity_forward(c[c["strike"] >= 5200.0], R, TAU) == pytest.approx(F, abs=1e-9)  # no strike near F


def test_parity_forward_uses_the_strike_where_call_and_put_are_closest():
    c = pd.DataFrame({"strike": [4900.0, 4900.0, 5000.0, 5000.0, 5100.0],
                      "cp": ["c", "p", "c", "p", "c"],
                      "mid": [130.0, 28.0, 62.0, 60.0, 20.0]})
    assert parity_forward(c, R, TAU) == pytest.approx(5000.0 + math.exp(R * TAU) * 2.0)


def test_parity_forward_needs_a_call_and_a_put_at_one_strike():
    c = pd.DataFrame({"strike": [4900.0, 5000.0, 5000.0], "cp": ["p", "c", "p"],
                      "mid": [28.0, 62.0, float("nan")]})
    assert math.isnan(parity_forward(c, R, TAU))
    assert math.isnan(parity_forward(c[c["cp"] == "p"], R, TAU))


def test_chain_columns_recover_the_inputs():
    chain = build_chain(quotes(), R)
    assert len(chain) == len(quotes())
    assert chain["tau"].unique() == pytest.approx([TAU])
    assert chain["F_parity"].unique() == pytest.approx([F])
    assert (chain["F"] == chain["F_parity"]).all() and (chain["F_fut"] == 5000.0).all()
    assert chain["forward_gap"].unique() == pytest.approx([3.0])
    assert chain["iv"].to_numpy() == pytest.approx(SIGMA, rel=1e-5)
    assert chain["moneyness"].to_numpy() == pytest.approx(np.log(chain["strike"] / F))
    row = chain[(chain["strike"] == 5100.0) & (chain["cp"] == "p")].iloc[0]
    assert row["mid"] == pytest.approx(price(F, 5100.0, TAU, R, SIGMA, "p"))
    expected = greeks(F, 5100.0, TAU, R, SIGMA, "p")
    assert {k: row[k] for k in expected} == pytest.approx(expected, rel=1e-4)


def test_using_the_futures_mid_instead_would_bias_implied_vols():
    # The futures mid is 3 points below the forward the options were priced on: calls look dear, puts cheap.
    chain = build_chain(quotes(), R)
    atm = chain[chain["strike"] == 5000.0].set_index("cp")
    from esvrp.pricing.black76 import implied_vol

    call_iv = implied_vol(atm.loc["c", "mid"], 5000.0, 5000.0, TAU, R, "c")
    put_iv = implied_vol(atm.loc["p", "mid"], 5000.0, 5000.0, TAU, R, "p")
    assert call_iv > SIGMA + 0.002 and put_iv < SIGMA - 0.002
    assert atm["iv"].to_numpy() == pytest.approx(SIGMA, rel=1e-5)


def test_each_chain_gets_its_own_forward():
    later = pd.Timestamp("2026-10-30 20:00", tz="UTC")
    q = pd.concat([quotes(F=5003.0), quotes(F=5010.0, expiry=later, sigma=0.2)], ignore_index=True)
    gaps = forward_gap_table(build_chain(q, R))
    assert list(gaps["expiry"]) == [EXPIRY, later]
    assert gaps["F_parity"].to_numpy() == pytest.approx([5003.0, 5010.0])
    assert gaps["forward_gap"].to_numpy() == pytest.approx([3.0, 10.0])


def test_one_sided_quotes_are_kept_without_mid_or_iv():
    q = quotes()
    q.loc[0, ["ask", "one_sided"]] = [float("nan"), True]
    chain = build_chain(q, R)
    assert len(chain) == len(q)
    assert np.isnan(chain.loc[0, "mid"]) and np.isnan(chain.loc[0, "iv"]) and np.isnan(chain.loc[0, "delta"])
    assert chain.loc[1:, "iv"].notna().all()


def test_falls_back_to_the_futures_mid_without_a_parity_forward():
    calls = quotes()[lambda q: q["cp"] == "c"]
    chain = build_chain(calls, R)
    assert chain["F_parity"].isna().all() and chain["forward_gap"].isna().all()
    assert (chain["F"] == 5000.0).all()
    assert chain["iv"].notna().any()


def test_rate_series_and_settlements_are_attached():
    r = risk_free_rate(pd.Index([DAY]), "constant", 0.05)
    settle = pd.DataFrame({"date": [DAY], "underlying": ["ESZ6"], "settle": [5001.5], "close": [5002.0]})
    chain = build_chain(quotes(), r, settle)
    assert (chain["r"] == 0.05).all() and (chain["settle"] == 5001.5).all()
    assert "close" not in chain
    assert build_chain(quotes(), R)["settle"].isna().all()
    with pytest.raises(NotImplementedError, match="fred"):
        risk_free_rate(pd.Index([DAY]), "fred", 0.05)


def test_forward_gap_figure_is_written(tmp_path):
    from esvrp.reporting.figures import forward_gap_figure

    later = pd.Timestamp("2026-11-30 21:00", tz="UTC")
    q = pd.concat([quotes(F=5003.0), quotes(F=4999.0, expiry=later)], ignore_index=True)
    out = forward_gap_figure(forward_gap_table(build_chain(q, R)), tmp_path / "figures" / "gap.png")
    assert out.exists() and out.stat().st_size > 10_000
    empty = forward_gap_figure(forward_gap_table(build_chain(q, R)).iloc[:0], tmp_path / "empty.png")
    assert empty.exists()
