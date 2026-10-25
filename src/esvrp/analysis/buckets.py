"""Moneyness, maturity and regime buckets (no look-ahead).

Implements PLANNING.md §4.5 (Moneyness and regimes) and §6 output 4 (the H2 table).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from esvrp.analysis.inference import Estimate, mean_with_se
from esvrp.hedging.costs import ES_MULTIPLIER


def regime_labels(
    series: pd.Series, window: str = "expanding", n_bins: int = 3, min_periods: int = 21,
) -> pd.Series:
    """Label each date by expanding-window percentile of information available at t only.

    The percentile of x_t is the share of observations up to and including t that are <= x_t. Labels
    run from 0 (lowest bin) to n_bins - 1, so with 3 bins a date is "high" (2) if x_t is above two
    thirds of the history so far. Dates with a missing value or fewer than `min_periods` observations
    of history get NaN. Appending later data never changes an earlier label.
    """
    if window != "expanding":
        raise ValueError(f"only window='expanding' is supported, got {window!r}")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    values = series.to_numpy(dtype=float)
    labels = np.full(len(values), np.nan)
    seen: list[float] = []
    for i, v in enumerate(values):
        if not np.isfinite(v):
            continue
        seen.append(v)
        if len(seen) < min_periods:
            continue
        pct = np.mean(np.asarray(seen) <= v)
        labels[i] = np.ceil(pct * n_bins) - 1
    return pd.Series(labels, index=series.index, name=series.name)


MONEYNESS_BUCKETS = ("otm_put", "near_atm", "otm_call")
REGIME_NAMES = {3: ("low", "mid", "high")}
BUCKET_COLUMNS = ["moneyness", "maturity", "regime", "measure", "n", "n_dates", "mean", "se_naive", "t_naive",
                  "se_hac", "t_hac", "hac_lags", "se_cluster", "t_cluster"]


def sw_by_date(vrp: pd.DataFrame, target_days: float = 30.0, sw_col: str = "sw") -> pd.Series:
    """One swap rate per date: the chain whose maturity is closest to `target_days`.

    The swap rate is observed at t, so labelling a date by it uses no later information.
    """
    v = vrp.dropna(subset=[sw_col]).assign(_gap=lambda d: (d["tau"] * 365 - target_days).abs())
    v = v.sort_values(["date", "_gap"]).drop_duplicates("date")
    return v.set_index("date")[sw_col].sort_index().rename("sw_level")


def moneyness_bucket(m: pd.Series, atm_width: float = 0.025) -> pd.Series:
    """OTM puts below -atm_width, near-ATM within it, OTM calls above (m = ln(K/F))."""
    m = m.astype(float)
    out = pd.Series(np.where(m < -atm_width, "otm_put", np.where(m > atm_width, "otm_call", "near_atm")),
                    index=m.index)
    return out.where(m.notna())


def bucket_table(
    hedged_gains: pd.DataFrame,
    vrp: pd.DataFrame,
    hedge_delta: str = "iv_daily",
    atm_width: float = 0.025,
    maturity_days: tuple[float, ...] = (14, 30, 60),
    n_bins: int = 3,
    min_periods: int = 21,
) -> pd.DataFrame:
    """H2: mean long delta-hedged gain by moneyness x maturity x regime (PLANNING.md §4.5, output 4).

    Two measures, before costs: `per_vega` (dollar gain over entry vega per vol point, in vol points, the
    units of the strategy's `gross_per_vega`) and `gain_over_F` (Bakshi-Kapadia). Each bucket also has a
    regime "all" row that pools the regimes, including dates still in the regime warm-up.

    Standard errors (§4.6): `naive` and `cluster` (by entry date) on the option-level gains; `hac` on the
    series of daily mean gains, with lags equal to the bucket's longest maturity in trading days minus
    one, since gains entered within that window overlap. `mean` is the option-level mean.

    Regimes are expanding-window terciles of the swap rate on the entry date (`sw_by_date`), so a label
    uses only information available at entry.
    """
    g = hedged_gains[hedged_gains["hedge_delta"] == hedge_delta].copy()
    g["date"] = pd.to_datetime(g["date"])
    g["per_vega"] = g["gain"] / (g["vega"] * ES_MULTIPLIER * 0.01)
    g["moneyness_bucket"] = moneyness_bucket(g["moneyness"], atm_width)
    edges = list(maturity_days)
    # Whole days: (14, 30, 60) gives "14-30d" and "31-60d"; the first bucket includes its lower edge.
    pairs = zip(edges[:-1], edges[1:], strict=True)
    labels = [f"{lo + (i > 0):g}-{hi:g}d" for i, (lo, hi) in enumerate(pairs)]
    days = g["tau"] * 365
    g["maturity_bucket"] = pd.cut(days, [edges[0] - 1e-9, *edges[1:]], labels=labels).astype(object)

    sw = vrp.assign(date=pd.to_datetime(vrp["date"]))
    regime = regime_labels(sw_by_date(sw), n_bins=n_bins, min_periods=min_periods)
    names = REGIME_NAMES.get(n_bins, tuple(str(i) for i in range(n_bins)))
    g["regime"] = g["date"].map(regime).map(lambda x: names[int(x)] if np.isfinite(x) else np.nan)

    hi_days = dict(zip(labels, edges[1:], strict=True))
    rows = []
    for (mb, tb), cell in g.dropna(subset=["moneyness_bucket", "maturity_bucket"]).groupby(
            ["moneyness_bucket", "maturity_bucket"]):
        lags = max(1, int(round(hi_days[tb] * 252 / 365)) - 1)
        for reg in ("all", *names):
            sub = cell if reg == "all" else cell[cell["regime"] == reg]
            for measure in ("per_vega", "gain_over_F"):
                rows.append({"moneyness": mb, "maturity": tb, "regime": reg, "measure": measure,
                             **_cell_stats(sub, measure, lags)})
    out = pd.DataFrame(rows, columns=BUCKET_COLUMNS)
    order = {k: i for i, k in enumerate(
        [*MONEYNESS_BUCKETS, *labels, "all", *names, "per_vega", "gain_over_F"])}
    return out.sort_values(["measure", "moneyness", "maturity", "regime"], key=lambda s: s.map(order),
                           ).reset_index(drop=True)


def _cell_stats(sub: pd.DataFrame, measure: str, lags: int) -> dict:
    x = sub[measure].astype(float)
    keep = x.notna()
    x, dates = x[keep], sub.loc[keep, "date"]
    naive = mean_with_se(x.to_numpy(), "naive")
    daily = x.groupby(dates).mean().sort_index()
    hac = mean_with_se(daily.to_numpy(), "hac", lags=min(lags, max(len(daily) - 1, 1)))
    cl = (mean_with_se(x.to_numpy(), "cluster", groups=dates.to_numpy()) if dates.nunique() > 1
          else Estimate(naive.mean, np.nan, np.nan, naive.n, "cluster"))
    return {"n": naive.n, "n_dates": int(dates.nunique()), "mean": naive.mean, "se_naive": naive.se,
            "t_naive": naive.t, "se_hac": hac.se, "t_hac": hac.t, "hac_lags": lags, "se_cluster": cl.se,
            "t_cluster": cl.t}
