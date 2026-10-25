"""Chain-quality and synchronization diagnostics.

Implements PLANNING.md §6 stages 2-4 (validation between quote collection and chain construction), §4.1,
§4.3 and §5.3.

The row rules of stage 2 decide whether a single quote is usable. This module decides whether the cleaned
quotes of one chain (one expiry on one day) are collectively usable: enough two-sided calls and puts, strikes
that reach far enough from the futures price, quotes that are fresh at the snapshot, and an option side
that is synchronous with the futures observation. A short strike range biases the variance-swap rate
downward, and stale or non-synchronous quotes add noise to implied volatilities and the parity forward.

`chain_quality` writes one row per chain with every diagnostic and the reasons for any failure, so no chain
disappears without a record. `quality.action` chooses between `flag` (keep everything, mark the
failures) and `reject` (drop failing chains before stage 4 builds them).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

KEYS = ["date", "expiry"]
COLUMNS = ["date", "expiry", "underlying", "F", "n_quotes", "n_calls", "n_puts", "n_strikes", "n_both_sides",
           "strike_min", "strike_max", "log_range_lo", "log_range_hi", "age_median", "age_p90", "age_max",
           "stale_share", "F_age_s", "sync_gap_s"]
FLOAT_COLUMNS = ["F", "strike_min", "strike_max", "log_range_lo", "log_range_hi", "age_median", "age_p90",
                 "age_max", "stale_share", "F_age_s", "sync_gap_s"]
REASONS = ["few_calls", "few_puts", "few_strikes", "narrow_range", "stale_quotes", "stale_futures",
           "unsynchronized"]


def _forward_col(quotes: pd.DataFrame) -> str:
    return "F_fut" if "F_fut" in quotes else "F"


def chain_quality(quotes: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """One row per chain (date, expiry) with diagnostics, `reasons` (semicolon-joined) and `status`.

    `quotes` are cleaned stage 2 quotes (or the stage 4 chain), with `quote_age_s`, the futures price `F`
    (`F_fut` in the chain) and, if available, `F_age_s`. `cfg` is the `quality` config section.

    Diagnostics: two-sided `n_calls` / `n_puts`, distinct `n_strikes`, `n_both_sides` (strikes with a
    two-sided call and put), `strike_min` / `strike_max`, `log_range_lo` / `log_range_hi` (ln(K/F) at the
    ends), quote age `age_median` / `age_p90` / `age_max` / `stale_share` (seconds, against
    `max_quote_age_s`), the futures age `F_age_s` and `sync_gap_s` (median |option quote time - futures
    observation time|). Without `F_age_s` the two futures diagnostics are empty and `sync_measured` is
    False; the chain is then not judged on them.
    `status` is `ok`, `flag` or `reject` (the latter only with `action: reject`).
    """
    f_col = _forward_col(quotes)
    q = quotes.copy()
    one_sided = q["one_sided"].astype(bool) if "one_sided" in q else q["ask"].isna()
    q["two_sided"] = ~one_sided
    q["call"] = q["two_sided"] & (q["cp"] == "c")
    q["put"] = q["two_sided"] & (q["cp"] == "p")
    q["stale"] = q["quote_age_s"] > cfg.max_quote_age_s
    has_sync = "F_age_s" in q
    if has_sync:
        q["sync_gap"] = (q["quote_age_s"] - q["F_age_s"]).abs()

    rows = []
    for key, g in q.groupby(KEYS, sort=True):
        F = g[f_col].iloc[0]
        calls, puts = set(g.loc[g["call"], "strike"]), set(g.loc[g["put"], "strike"])
        age = g["quote_age_s"]
        row = {
            "date": key[0], "expiry": key[1], "underlying": g["underlying"].iloc[0], "F": F,
            "n_quotes": len(g), "n_calls": int(g["call"].sum()), "n_puts": int(g["put"].sum()),
            "n_strikes": g["strike"].nunique(), "n_both_sides": len(calls & puts),
            "strike_min": g["strike"].min(), "strike_max": g["strike"].max(),
            "log_range_lo": np.log(g["strike"].min() / F), "log_range_hi": np.log(g["strike"].max() / F),
            "age_median": age.median(), "age_p90": age.quantile(0.9), "age_max": age.max(),
            "stale_share": g["stale"].mean(),
            "F_age_s": g["F_age_s"].iloc[0] if has_sync else np.nan,
            "sync_gap_s": g["sync_gap"].median() if has_sync else np.nan,
        }
        rows.append(row)
    out = pd.DataFrame(rows, columns=COLUMNS)  # fixed columns, so an empty input still has the full schema

    out = out.astype(dict.fromkeys(FLOAT_COLUMNS, float))
    out["sync_measured"] = out["F_age_s"].notna()
    fails = {
        "few_calls": out["n_calls"] < cfg.min_calls,
        "few_puts": out["n_puts"] < cfg.min_puts,
        "few_strikes": out["n_strikes"] < cfg.min_strikes,
        "narrow_range": ((out["strike_min"] > out["F"] * (1 - cfg.min_range_pct))
                         | (out["strike_max"] < out["F"] * (1 + cfg.min_range_pct))),
        "stale_quotes": out["stale_share"] > cfg.max_stale_share,
        "stale_futures": out["F_age_s"] > cfg.max_futures_age_s,
        "unsynchronized": out["sync_gap_s"] > cfg.max_sync_gap_s,
    }
    for name in REASONS:
        out[f"fail_{name}"] = fails[name].fillna(False).astype(bool)
    out["reasons"] = out[[f"fail_{r}" for r in REASONS]].apply(
        lambda row: ";".join(r for r, bad in zip(REASONS, row, strict=True) if bad), axis=1)
    failing = out["reasons"] != ""
    out["status"] = np.where(failing, "reject" if cfg.action == "reject" else "flag", "ok")
    return out


def quote_age_by_day(quotes: pd.DataFrame, max_quote_age_s: float) -> pd.DataFrame:
    """Quote-age distribution per day over all chains: count, mean, quantiles, max and share stale."""
    g = quotes.groupby("date")["quote_age_s"]
    out = pd.DataFrame({
        "n_quotes": g.size(), "mean": g.mean(), "median": g.median(), "p90": g.quantile(0.9),
        "p99": g.quantile(0.99), "max": g.max(),
        "stale_share": g.apply(lambda a: float((a > max_quote_age_s).mean())),
    })
    return out.reset_index()


def reason_summary(quality: pd.DataFrame) -> pd.DataFrame:
    """Chains failing each rule (one chain can fail several), plus totals, as a two-column table."""
    counts = {"chains": len(quality), "failing": int((quality["reasons"] != "").sum()),
              "rejected": int((quality["status"] == "reject").sum()),
              "sync_not_measured": int((~quality["sync_measured"]).sum())}
    for name in REASONS:
        counts[name] = int(quality[f"fail_{name}"].sum())
    return pd.Series(counts, name="chains").rename_axis("rule").reset_index()


def apply_quality(quotes: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Drop the quotes of chains with status `reject`; the reasons stay in the `quality` table."""
    rejected = quality.loc[quality["status"] == "reject", KEYS]
    if rejected.empty:
        return quotes
    marked = quotes.merge(rejected.assign(_rejected=True), on=KEYS, how="left")
    return quotes[marked["_rejected"].isna().to_numpy()].reset_index(drop=True)
