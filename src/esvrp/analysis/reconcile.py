"""Reconciliation of the variance risk premium with delta-hedged gains (H3).

Implements PLANNING.md §4.5 (Why it measures the premium) and roadmap 15.

H3: the VRP (`RV - SW`, Carr-Wu, stage 5) and the gamma-weighted hedged gains (stage 6) agree in sign and
co-move. Both are in annualized variance units:

- Hedging at a fixed implied vol `sigma_i`, `Pi ~ 1/2 sum Gamma_n F_n^2 (r_{n+1}^2 - sigma_i^2 dt_n)`, so
  for each chain (one expiry on one day) `HG = sum Pi / sum 1/2 Gamma_n F_n^2 dt_n` over its options is a
  gamma-weighted average of realized minus implied variance. Summing before dividing matters: averaging
  each option's ratio is biased upward by about 0.0015 at daily hedging (an 18% vol reads as 18.4%).
- `HG = RV_gw - IV2_gw` with `IV2_gw` the gamma-weighted implied variance of the chain's options, so
  `HG - VRP = (RV_gw - RV) - (IV2_gw - SW)`. The second gap is the skew: the strip weights every strike
  by 1/K^2, while gamma weighting concentrates near the money, where implied vol is lowest. A
  less negative HG than VRP is therefore expected, not a failure of H3.

This is the standard Black-Scholes hedging identity, not a result of either source paper; Bakshi and
Kapadia's own link is vega-weighted (§4.5). H3 tests agreement in sign and co-movement, not equality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from esvrp.analysis.inference import mean_with_se
from esvrp.hedging.costs import ES_MULTIPLIER

TZ = "America/Chicago"
KEYS = ["date", "expiry"]


def chain_hedged_variance(hedged_gains: pd.DataFrame, hedge_delta: str = "iv_fixed",
                          moneyness_max: float | None = None) -> pd.DataFrame:
    """Gamma-weighted hedged gain per chain, in variance units, from the stage 6 gains.

    `iv_fixed` by default: the identity above holds for a hedge at the option's own entry implied vol.
    `moneyness_max` narrows the options used (default: all of stage 6, |moneyness| <= 10%).
    Columns: `hg` (sum Pi / sum weight), `iv2_gw` (gamma-weighted implied variance), `rv_gw = hg + iv2_gw`,
    `n_options` and `tau`.
    """
    g = hedged_gains[(hedged_gains["hedge_delta"] == hedge_delta) & (hedged_gains["gamma_dt"] > 0)]
    if moneyness_max is not None:
        g = g[g["moneyness"].abs() <= moneyness_max]
    g = g.assign(pi=g["gain"] / ES_MULTIPLIER, w_iv2=g["gamma_dt"] * g["iv"] ** 2)
    out = g.groupby(KEYS).agg(pi=("pi", "sum"), weight=("gamma_dt", "sum"), w_iv2=("w_iv2", "sum"),
                              n_options=("pi", "size"), tau=("tau", "first"),
                              underlying=("underlying", "first"))
    out["hg"] = out["pi"] / out["weight"]
    out["iv2_gw"] = out["w_iv2"] / out["weight"]
    out["rv_gw"] = out["hg"] + out["iv2_gw"]
    return out[["underlying", "tau", "n_options", "hg", "iv2_gw", "rv_gw"]].reset_index()


def monthly_subsample(df: pd.DataFrame) -> pd.DataFrame:
    """One chain per end-of-month expiry: the first date in the expiry's own month (non-overlapping).

    The same entry rule as the strategy backtest, so consecutive observations do not share any days.
    """
    month = pd.to_datetime(df["date"]).dt.to_period("M")
    expiry = pd.to_datetime(df["expiry"], utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    expiry_month = expiry.dt.to_period("M")
    d = df[month == expiry_month]
    return d[d["date"] == d.groupby("expiry")["date"].transform("min")].reset_index(drop=True)


def reconcile(vrp: pd.DataFrame, hedged: pd.DataFrame, vrp_col: str = "vrp_diff", hac_lags: int = 30,
              block_days: int = 21, n_boot: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join the VRP and hedged-gain series per chain and test H3.

    `vrp` is stage 5's `vrp.parquet` (one row per chain, matched maturity); `hedged` is
    `chain_hedged_variance`. Returns the joined chains and a table with one row for every chain
    (overlapping daily starts) and one for the non-overlapping monthly subsample:

    - `mean_vrp`, `mean_hg` with naive and HAC t-statistics (HAC lag 30 on the daily series, Carr-Wu;
      6 on the monthly one, §4.6);
    - `sign_agree`: share of chains where the two have the same sign;
    - `pearson` with a 95% interval: a block bootstrap over dates (blocks of `block_days` trading days,
      so the overlap between windows is kept) for the daily series, Fisher's z for the monthly one;
    - `spearman`, the rank correlation.
    """
    cols = [*KEYS, vrp_col, *(c for c in ("sw", "rv") if c in vrp.columns)]
    j = vrp[cols].merge(hedged, on=KEYS, how="inner").rename(columns={vrp_col: "vrp"})
    j = j[np.isfinite(j["vrp"]) & np.isfinite(j["hg"])].sort_values(KEYS).reset_index(drop=True)
    monthly = monthly_subsample(j)
    rows = {"all chains (overlapping)": _h3_stats(j, hac_lags, block_days, n_boot, seed),
            "monthly (non-overlapping)": _h3_stats(monthly, min(6, hac_lags), None, n_boot, seed)}
    return j, pd.DataFrame(rows).T


def _h3_stats(j: pd.DataFrame, lags: int, block_days: int | None, n_boot: int, seed: int) -> dict:
    n = len(j)
    out = {"n": n, "n_dates": j["date"].nunique() if n else 0}
    if n < 3:
        return out
    x, y = j["vrp"].to_numpy(float), j["hg"].to_numpy(float)
    for name, v in (("vrp", x), ("hg", y)):
        out[f"mean_{name}"] = float(v.mean())
        out[f"t_{name}_naive"] = mean_with_se(v, "naive").t
        out[f"t_{name}_hac"] = mean_with_se(v, "hac", lags=min(lags, n - 2)).t
    out["sign_agree"] = float((np.sign(x) == np.sign(y)).mean())
    r = float(np.corrcoef(x, y)[0, 1])
    out["pearson"] = r
    out["spearman"] = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    if block_days is None:
        out["ci_low"], out["ci_high"] = _fisher_ci(r, n)
        out["ci_method"] = "Fisher z"
    else:
        out["ci_low"], out["ci_high"] = _date_block_bootstrap_ci(j, block_days, n_boot, seed)
        out["ci_method"] = f"date-block bootstrap ({block_days} days)"
    return out


def _fisher_ci(r: float, n: int, z_crit: float = 1.959964) -> tuple[float, float]:
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return float("nan"), float("nan")
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se))


def _date_block_bootstrap_ci(j: pd.DataFrame, block: int, n_boot: int, seed: int,
                             level: float = 0.95) -> tuple[float, float]:
    """Percentile interval for the Pearson correlation, resampling blocks of consecutive dates.

    All chains on a date move together, so whole dates are resampled, in runs of `block` dates to keep
    the dependence between overlapping windows.
    """
    dates = np.sort(j["date"].unique())
    if len(dates) < 2:
        return float("nan"), float("nan")
    rows_by_date = j.groupby("date").indices
    by_pos = [rows_by_date[d] for d in dates]
    x, y = j["vrp"].to_numpy(float), j["hg"].to_numpy(float)
    block = max(1, min(block, len(dates)))
    n_blocks = int(np.ceil(len(dates) / block))
    block_rows = [np.concatenate(by_pos[s:s + block]) for s in range(len(dates) - block + 1)]
    rng = np.random.default_rng(seed)
    stats = []
    for starts in rng.integers(0, len(block_rows), size=(n_boot, n_blocks)):
        idx = np.concatenate([block_rows[s] for s in starts])
        if np.std(x[idx]) > 0 and np.std(y[idx]) > 0:
            stats.append(np.corrcoef(x[idx], y[idx])[0, 1])
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.quantile(stats, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)
