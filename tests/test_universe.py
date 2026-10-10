"""Tests for data/universe.py (PLANNING.md §6 stage 1, roadmap 4). Fixture definitions, no network."""

import pandas as pd
import pytest

from esvrp.data.universe import build_universe, check_strikes_bracket_futures, pull_universe


def make_defs():
    """Shaped like Databento definition records for the EW.OPT parent."""
    return pd.DataFrame({
        "instrument_id": [1, 2, 3, 4, 5, 2],
        "raw_symbol": ["EWV6 C5000", "EWV6 P5000", "EWV6 C5100", "EWV6-EWX6", "ESZ6 C5000", "EWV6 P5000"],
        "instrument_class": ["C", "P", "C", "T", "C", "P"],
        "strike_price": [5000.0, 5000.0, 5100.0, float("nan"), 5000.0, 5000.0],
        "expiration": pd.to_datetime(
            ["2026-10-30 20:00"] * 4 + ["2026-12-18 14:30", "2026-10-30 20:00"], utc=True
        ),
        "underlying": ["ESZ6", "ESZ6", "ESZ6", "", "ESZ6", "ESZ6"],
        "cfi": ["OCEFPS", "OPEFPS", "OCEFPS", "OMXFPS", "OCAFPS", "OPEFPS"],
        "user_defined_instrument": ["N", "N", "Y", "Y", "N", "N"],
    })


def test_definitions_parse_to_universe():
    uni = build_universe(make_defs())
    assert list(uni["instrument_id"]) == [1, 3, 2]  # sorted by expiry, cp, strike
    assert set(uni["cp"]) == {"c", "p"}
    assert (uni["underlying"] == "ESZ6").all()
    assert (uni["style"] == "european").all()
    assert uni.loc[uni["instrument_id"] == 3, "user_defined"].item()


def test_spreads_and_duplicates_dropped():
    uni = build_universe(make_defs())
    assert 4 not in set(uni["instrument_id"])          # spread (instrument_class T)
    assert uni["instrument_id"].is_unique             # repeated definition of id 2


def test_american_dropped_unless_requested():
    assert 5 not in set(build_universe(make_defs())["instrument_id"])
    both = build_universe(make_defs(), european_only=False)
    assert both.loc[both["instrument_id"] == 5, "style"].item() == "american"


def test_missing_underlying_raises():
    defs = make_defs()
    defs.loc[0, "underlying"] = ""
    with pytest.raises(ValueError, match="no underlying"):
        build_universe(defs)


def test_strikes_bracket_futures():
    uni = build_universe(make_defs())
    check_strikes_bracket_futures(uni, 5050.0)
    check_strikes_bracket_futures(uni, 5050.0, underlying="ESZ6")
    with pytest.raises(ValueError, match="do not bracket"):
        check_strikes_bracket_futures(uni, 505.0)          # strikes off by a power of ten
    with pytest.raises(ValueError, match="no strikes"):
        check_strikes_bracket_futures(uni, 5050.0, underlying="ESH7")


def test_pull_universe_unions_weekly_snapshots(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import esvrp.data.universe as universe

    calls = []

    def fake_fetch(dataset, schema, symbols, start, end, budget_usd, stype_in, client):
        calls.append((schema, symbols, stype_in, str(start)))
        defs = make_defs()  # id 1 only exists in the first week's snapshot (pulls run in parallel)
        return defs if str(start) == "2026-09-01" else defs[defs["instrument_id"] != 1]

    monkeypatch.setattr(universe, "fetch", fake_fetch)
    cfg = SimpleNamespace(data=SimpleNamespace(dataset="GLBX.MDP3", series="monthly_eom",
                                               start="2026-09-01", end="2026-09-30", budget_usd=50))
    uni = pull_universe(cfg)
    assert {c[:3] for c in calls} == {("definition", "EW.OPT", "parent")}
    assert len(calls) == 5                                  # Tuesdays in Sep 2026
    assert set(uni["instrument_id"]) == {1, 2, 3}           # union keeps id 1 from week one
