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
    far = chain["strike"] < 3000
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
    gappy = chain[~chain["strike"].between(3000, 3100, inclusive="neither")]  # a zero-bid hole
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
