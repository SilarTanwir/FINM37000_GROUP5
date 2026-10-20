"""Tests for data/quality.py (PLANNING.md §6 stages 2-4, roadmap 20). Fixtures only, no network."""

from types import SimpleNamespace

import pandas as pd

from esvrp.config import load_config
from esvrp.data.quality import apply_quality, chain_quality, quote_age_by_day, reason_summary

F = 5000.0
CFG = SimpleNamespace(action="flag", min_calls=3, min_puts=3, min_strikes=6, min_range_pct=0.10,
                      max_quote_age_s=300.0, max_stale_share=0.25, max_futures_age_s=300.0,
                      max_sync_gap_s=300.0)


def chain(expiry="2026-10-30", strikes=(4400, 4600, 4800, 5000, 5200, 5400, 5600), age=0.0, f_age=0.0,
          date="2026-09-01", **over):
    """A two-sided call and put at every strike, all quotes `age` seconds old."""
    rows = [{"date": date, "expiry": expiry, "underlying": "ESZ6", "cp": cp, "strike": float(k),
             "bid": 10.0, "ask": 11.0, "one_sided": False, "F": F, "quote_age_s": age, "F_age_s": f_age}
            for k in strikes for cp in ("c", "p")]
    return pd.DataFrame(rows).assign(**over)


def one(quotes, cfg=CFG):
    return chain_quality(quotes, cfg).iloc[0]


def test_complete_chain_passes_and_records_diagnostics():
    row = one(chain())
    assert row["status"] == "ok" and row["reasons"] == ""
    assert (row["n_calls"], row["n_puts"], row["n_strikes"], row["n_both_sides"]) == (7, 7, 7, 7)
    assert row["strike_min"] == 4400 and row["strike_max"] == 5600
    assert row["log_range_lo"] < 0 < row["log_range_hi"]
    assert row["age_median"] == 0 and row["stale_share"] == 0 and row["sync_gap_s"] == 0


def test_incomplete_strike_coverage_is_flagged_with_reasons():
    row = one(chain(strikes=(4900, 5000, 5100)))  # 3 strikes, within +-2% of F
    assert row["status"] == "flag"
    assert set(row["reasons"].split(";")) == {"few_strikes", "narrow_range"}


def test_one_sided_quotes_do_not_count_as_valid():
    q = chain()
    q.loc[q["cp"] == "p", ["ask", "one_sided"]] = [None, True]  # no two-sided puts
    row = one(q)
    assert row["n_puts"] == 0 and row["n_calls"] == 7
    assert row["reasons"] == "few_puts"


def test_missing_wing_on_one_side_fails_range():
    row = one(chain(strikes=(4400, 4600, 4800, 5000, 5100, 5200, 5300)))  # high end only +6%
    assert row["reasons"] == "narrow_range"


def test_stale_quotes_flagged_by_share():
    q = chain()
    q.loc[q["strike"] <= 4800, "quote_age_s"] = 600.0  # 6 of 14 quotes stale (43%)
    row = one(q)
    assert row["reasons"] == "stale_quotes"
    assert abs(row["stale_share"] - 6 / 14) < 1e-12 and row["age_max"] == 600.0
    assert one(q, SimpleNamespace(**{**vars(CFG), "max_stale_share": 0.5}))["status"] == "ok"


def test_futures_observation_gap_flagged():
    row = one(chain(f_age=900.0, age=900.0))  # futures 15 min old, options equally old: in sync but stale
    assert set(row["reasons"].split(";")) == {"stale_futures", "stale_quotes"}
    assert row["sync_gap_s"] == 0


def test_option_futures_desynchronization_flagged():
    row = one(chain(age=0.0, f_age=400.0, ))  # options at the snapshot, futures 400 s earlier
    assert set(row["reasons"].split(";")) == {"stale_futures", "unsynchronized"}
    assert row["sync_gap_s"] == 400.0


def test_unmeasured_synchronization_is_recorded_not_failed():
    row = one(chain().drop(columns="F_age_s"))
    assert row["status"] == "ok" and not row["sync_measured"] and pd.isna(row["sync_gap_s"])


def test_each_chain_judged_separately_and_reject_mode_drops_only_failures():
    quotes = pd.concat([chain("2026-10-30"), chain("2026-11-30", strikes=(4900, 5000, 5100))])
    flagged = chain_quality(quotes, CFG)
    assert dict(zip(flagged["expiry"], flagged["status"], strict=True)) == {"2026-10-30": "ok",
                                                                             "2026-11-30": "flag"}
    assert len(apply_quality(quotes, flagged)) == len(quotes)  # flag keeps everything

    rejected = chain_quality(quotes, SimpleNamespace(**{**vars(CFG), "action": "reject"}))
    kept = apply_quality(quotes, rejected)
    assert set(kept["expiry"]) == {"2026-10-30"} and len(kept) == 14
    assert rejected.loc[rejected["status"] == "reject", "reasons"].iloc[0] != ""  # reason retained


def test_summaries_count_reasons_and_age_by_day():
    quotes = pd.concat([chain("2026-10-30"), chain("2026-11-30", strikes=(4900, 5000, 5100), age=600.0)])
    quality = chain_quality(quotes, CFG)
    s = reason_summary(quality).set_index("rule")["chains"]
    assert (s["chains"], s["failing"], s["few_strikes"], s["stale_quotes"], s["rejected"]) == (2, 1, 1, 1, 0)
    by_day = quote_age_by_day(quotes, 300.0).iloc[0]
    assert by_day["n_quotes"] == 14 + 6 and by_day["max"] == 600.0
    assert abs(by_day["stale_share"] - 6 / 20) < 1e-12


def test_default_config_has_quality_section():
    q = load_config().quality
    assert q.action in {"flag", "reject"} and q.min_strikes >= q.min_calls > 0
