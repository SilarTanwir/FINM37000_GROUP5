"""PLANNING.md §9: test_pipeline_smoke (end-to-end on synthetic data)."""

import pytest

from esvrp.synthetic import simulate_chain, simulate_underlying

pytestmark = pytest.mark.xfail(reason="not implemented", strict=False)


def test_pipeline_produces_expected_outputs(tmp_path):
    path = simulate_underlying(n_days=60, sigma=0.18)
    chain = simulate_chain(5000.0, 0.18, 30 / 365, 0.04, [4800.0, 5000.0, 5200.0])
    assert len(path) > 0 and len(chain) > 0
    expected = ["universe", "quotes", "underlying_5m", "underlying_daily", "chain", "strip",
                "rv", "vrp", "hedged_gains"]
    raise NotImplementedError(f"run stages 1-8 into {tmp_path}; check {expected}")
