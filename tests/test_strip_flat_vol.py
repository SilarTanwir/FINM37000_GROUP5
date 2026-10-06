"""PLANNING.md §9: test_strip_flat_vol."""

import pytest

from esvrp.vrp.strip import variance_swap_rate

pytestmark = pytest.mark.xfail(reason="not implemented", strict=False)


def test_flat_vol_recovers_sigma_squared(flat_vol_chain):
    chain, F, r, tau = flat_vol_chain(sigma=0.2)
    res = variance_swap_rate(chain, F, r, tau)
    assert res.sw == pytest.approx(0.2**2, rel=0.01)


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
    assert sw_narrow.truncation_flag
