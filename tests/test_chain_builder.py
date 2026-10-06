"""PLANNING.md §9: test_chain_builder (fixture definitions and cleaning rules)."""

import pandas as pd
import pytest

from esvrp.data.quotes import clean_quotes
from esvrp.data.universe import build_universe

pytestmark = pytest.mark.xfail(reason="not implemented", strict=False)


def test_definitions_parse_to_universe():
    defs = pd.DataFrame({
        "raw_symbol": ["ESZ6 C5000", "ESZ6 P5000"],
        "instrument_class": ["C", "P"],
        "strike_price": [5000.0, 5000.0],
        "expiration": pd.to_datetime(["2026-12-18", "2026-12-18"]),
        "underlying": ["ESZ6", "ESZ6"],
    })
    uni = build_universe(defs)
    assert set(uni["cp"]) == {"c", "p"}
    assert (uni["strike"] == 5000.0).all()
    assert (uni["underlying"] == "ESZ6").all()


def test_cleaning_rules_drop_bad_quotes():
    q = pd.DataFrame({
        "strike": [4900, 5000, 5100, 5200],
        "bid": [0.0, 10.0, 12.0, 5.0],   # zero bid
        "ask": [1.0, 11.0, 11.0, 6.0],   # crossed at 5100
    })
    cleaned, log = clean_quotes(q, cfg=None)
    assert list(cleaned["strike"]) == [5000, 5200]
    assert log["zero_bid"] == 1 and log["crossed"] == 1
