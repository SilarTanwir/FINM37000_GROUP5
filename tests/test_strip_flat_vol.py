"""PLANNING.md §9: test_strip_flat_vol (roadmap 9)."""

import math

import numpy as np
import pandas as pd
import pytest

from esvrp.pricing.black76 import price
from esvrp.vrp.strip import strip_table, variance_swap_rate


def test_flat_vol_recovers_sigma_squared(flat_vol_chain):
    chain, F, r, tau = flat_vol_chain(sigma=0.2)
    res = variance_swap_rate(chain, F, r, tau)
    assert res.sw == pytest.approx(0.2**2, rel=0.01)
    assert not res.truncation_flag


def test_error_shrinks_with_finer_grid(flat_vol_chain):
    errs = []
    for step in (100.0, 25.0, 5.0):
        chain, F, r, tau = flat_vol_chain(sigma=0.2, step=step)
        errs.append(abs(variance_swap_rate(chain, F, r, tau).sw - 0.2**2))
    assert errs[0] > errs[1] > errs[2]


def test_narrow_range_biases_downward(flat_vol_chain):
    wide, F, r, tau = flat_vol_chain(sigma=0.2)
    narrow, *_ = flat_vol_chain(sigma=0.2, lo=0.95, hi=1.05)
    sw_wide = variance_swap_rate(wide, F, r, tau)
    sw_narrow = variance_swap_rate(narrow, F, r, tau)
    assert sw_narrow.sw < sw_wide.sw
    assert sw_narrow.truncation_flag and not sw_wide.truncation_flag


def test_hand_computed_three_strikes():
    """F=100, r=0, tau=1, K0=100: 2 * [10/90^2*1 + 10/100^2*avg(3,2) + 10/110^2*1] with no correction."""
    chain = pd.DataFrame({
        "strike": [90.0, 100.0, 100.0, 110.0], "cp": ["p", "c", "p", "c"],
        "mid": [1.0, 3.0, 2.0, 1.0],
    })
    res = variance_swap_rate(chain, F=100.0, r=0.0, tau=1.0, min_range_pct=0.05)
    expected = 2 * (10 / 90**2 * 1.0 + 10 / 100**2 * 2.5 + 10 / 110**2 * 1.0)
    assert res.sw == pytest.approx(expected, rel=1e-12)
    assert res.k0 == 100.0 and res.n_strikes == 3 and not res.k0_one_sided
    assert res.contributions.sum() - res.correction == pytest.approx(res.sw)


def test_correction_term_when_forward_above_k0():
    chain = pd.DataFrame({
        "strike": [90.0, 100.0, 100.0, 110.0], "cp": ["p", "c", "p", "c"], "mid": [1.0, 3.0, 2.0, 1.0],
    })
    res = variance_swap_rate(chain, F=101.0, r=0.0, tau=0.5, min_range_pct=0.05)
    assert res.k0 == 100.0
    assert res.correction == pytest.approx((101 / 100 - 1) ** 2 / 0.5)


def test_discount_factor_scales_sw(flat_vol_chain):
    chain, F, r, tau = flat_vol_chain(sigma=0.2, r=0.04)
    sw = variance_swap_rate(chain, F, r, tau).sw
    no_disc = variance_swap_rate(chain, F, 0.0, tau).sw
    assert sw / no_disc == pytest.approx(math.exp(r * tau), rel=0.02)


def test_bid_mid_ask_ordering(flat_vol_chain):
    chain, F, r, tau = flat_vol_chain(sigma=0.2)
    chain = chain.assign(bid=chain["mid"] * 0.97, ask=chain["mid"] * 1.03)
    bid, mid, ask = (variance_swap_rate(chain, F, r, tau, c).sw for c in ("bid", "mid", "ask"))
    assert bid < mid < ask


def test_one_sided_quotes_only_enter_bid_variant(flat_vol_chain):
    chain, F, r, tau = flat_vol_chain(sigma=0.2)
    chain = chain.assign(bid=chain["mid"], ask=chain["mid"])
    far = chain["strike"] < 4500
    chain.loc[far, ["ask", "mid"]] = np.nan  # one-sided: bid only
    n_mid = variance_swap_rate(chain, F, r, tau, "mid").n_strikes
    n_bid = variance_swap_rate(chain, F, r, tau, "bid").n_strikes
    assert n_bid > n_mid


def test_missing_k0_side_is_flagged():
    chain = pd.DataFrame({"strike": [90.0, 100.0, 110.0], "cp": ["p", "c", "c"], "mid": [1.0, 3.0, 1.0]})
    res = variance_swap_rate(chain, F=100.0, r=0.0, tau=1.0)
    assert res.k0_one_sided and res.n_strikes == 3


def test_missing_strikes_widen_dk(flat_vol_chain):
    chain, F, r, tau = flat_vol_chain(sigma=0.2, step=10.0)
    gappy = chain[~chain["strike"].between(4400, 4500, inclusive="neither")]  # a zero-bid hole
    res = variance_swap_rate(gappy, F, r, tau)
    assert res.max_strike_gap > 10.0
    assert res.sw == pytest.approx(variance_swap_rate(chain, F, r, tau).sw, rel=0.02)


@pytest.mark.parametrize("F,tau", [(float("nan"), 0.1), (0.0, 0.1), (5000.0, 0.0), (5000.0, float("nan"))])
def test_invalid_inputs_give_nan(flat_vol_chain, F, tau):
    chain, *_ = flat_vol_chain()
    res = variance_swap_rate(chain, F, 0.04, tau)
    assert math.isnan(res.sw) and res.n_strikes == 0


def test_bad_price_col_raises(flat_vol_chain):
    chain, F, r, tau = flat_vol_chain()
    with pytest.raises(ValueError):
        variance_swap_rate(chain, F, r, tau, "last")


def test_strip_table_one_row_per_chain(flat_vol_chain):
    frames = []
    for day, sigma in [("2026-09-15", 0.15), ("2026-09-16", 0.25)]:
        chain, F, r, tau = flat_vol_chain(sigma=sigma)
        frames.append(chain.assign(date=day, expiry="2026-10-30", underlying="ESZ6", F=F, r=r, tau=tau,
                                   bid=chain["mid"], ask=chain["mid"]))
    table = strip_table(pd.concat(frames, ignore_index=True))
    assert len(table) == 2
    assert table["sw_mid"].to_numpy() == pytest.approx([0.15**2, 0.25**2], rel=0.01)
    assert table["sw_bid"].to_numpy() == pytest.approx(table["sw_mid"].to_numpy())


@pytest.mark.parametrize("F", [5000.0, 5003.0, 5009.0, 5024.0])
def test_sw_does_not_depend_on_where_f_falls_between_strikes(F):
    """The (F/K0 - 1)^2 term makes SW invariant to F vs K0 (Cboe formula); without it SW drifts."""
    tau, r, sigma = 30 / 365, 0.04, 0.2
    rows = [{"strike": float(k), "cp": cp, "mid": price(F, float(k), tau, r, sigma, cp)}
            for k in np.arange(2500, 8000, 5.0) for cp in "pc"]
    res = variance_swap_rate(pd.DataFrame(rows), F, r, tau)
    assert res.sw == pytest.approx(sigma**2, rel=1e-4)
    if F != res.k0:
        assert abs(res.sw - sigma**2) < abs(res.sw + res.correction - sigma**2)


def _with_junk_call(chain, F, r, tau, strike=10100.0, iv=5.08):
    junk = pd.DataFrame({"strike": [strike], "cp": ["c"], "mid": [price(F, strike, tau, r, iv, "c")]})
    junk["bid"] = junk["ask"] = junk["mid"]
    return pd.concat([chain, junk], ignore_index=True)


def test_junk_far_quote_is_capped_out(flat_vol_chain):
    """A single far-out call quoted at ~500% vol must not move the swap rate (Tanwir's EWQ6 C10100 case)."""
    chain, F, r, tau = flat_vol_chain(sigma=0.2, hi=1.3)
    clean = variance_swap_rate(chain, F, r, tau)
    dirty = _with_junk_call(chain, F, r, tau)
    capped = variance_swap_rate(dirty, F, r, tau)
    assert capped.sw == pytest.approx(clean.sw, rel=1e-9)
    assert capped.n_dropped_iv == clean.n_dropped_iv + 1
    uncapped = variance_swap_rate(dirty, F, r, tau, max_iv=None)
    assert uncapped.sw > 2 * clean.sw  # without the cap the junk quote dominates the sum


def test_cap_applies_to_every_variant_and_to_one_sided_quotes(flat_vol_chain):
    chain, F, r, tau = flat_vol_chain(sigma=0.2, hi=1.3)
    dirty = _with_junk_call(chain, F, r, tau)
    base = {c: variance_swap_rate(chain.assign(bid=chain["mid"], ask=chain["mid"]), F, r, tau, c).sw
            for c in ("bid", "mid", "ask")}
    dirty = dirty.assign(bid=dirty["mid"], ask=dirty["mid"])
    for col in ("bid", "mid", "ask"):
        assert variance_swap_rate(dirty, F, r, tau, col).sw == pytest.approx(base[col], rel=1e-9)
    one_sided = dirty.copy()
    one_sided.loc[one_sided["strike"] == 10100.0, ["ask", "mid"]] = np.nan  # a bid-only junk quote
    assert variance_swap_rate(one_sided, F, r, tau, "bid").sw == pytest.approx(base["bid"], rel=1e-9)


def test_chain_iv_column_is_used_when_present(flat_vol_chain):
    """With an `iv` column the cap follows it, so a quote with iv above the cap goes even at a fair price."""
    chain, F, r, tau = flat_vol_chain(sigma=0.2, hi=1.3)
    chain = chain.assign(iv=0.2)
    chain.loc[chain["strike"] == 5500.0, "iv"] = 3.0
    res = variance_swap_rate(chain, F, r, tau)
    assert 5500.0 not in res.contributions.index
    assert variance_swap_rate(chain, F, r, tau, max_iv=None).contributions.index.isin([5500.0]).any()


def test_iv_bounds_can_be_changed(flat_vol_chain):
    chain, F, r, tau = flat_vol_chain(sigma=0.2, hi=1.3)
    assert variance_swap_rate(chain, F, r, tau, min_iv=0.25).n_strikes == 0  # every quote is at 20% vol
    assert variance_swap_rate(chain, F, r, tau, max_iv=0.15).n_strikes == 0


def test_strip_table_reports_dropped_quotes(flat_vol_chain):
    from esvrp.vrp.strip import strip_table

    chain, F, r, tau = flat_vol_chain(sigma=0.2, hi=1.3)
    chain = _with_junk_call(chain, F, r, tau).assign(
        date="2026-09-15", expiry="2026-10-30", underlying="ESZ6", F=F, r=r, tau=tau)
    chain = chain.assign(bid=chain["mid"], ask=chain["mid"])
    row = strip_table(chain).iloc[0]
    assert row["n_dropped_iv"] >= 1 and row["sw_mid"] == pytest.approx(0.2**2, rel=0.01)
    assert strip_table(chain, max_iv=None).iloc[0]["sw_mid"] > 2 * row["sw_mid"]
