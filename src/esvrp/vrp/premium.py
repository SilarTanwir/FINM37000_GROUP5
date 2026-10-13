"""VRP series, constant-maturity interpolation and descriptives.

Implements PLANNING.md §4.3 (maturity handling) and §4.4.

`vrp_table` joins the strip (roadmap 9) and realized variance (roadmap 10) of each chain. The primary
premium is the difference RV - SW in variance units (decided, §15.6); ln(RV/SW) is reported alongside
for inference and RV/SW - 1 is the variance-swap excess return. A negative mean means variance buyers
pay a premium. SW is known at the snapshot and RV only at expiry, so a pair is observable at expiry.

`constant_maturity` follows Carr and Wu (2009), Eq. (10): expiries closer than `min_front_days` are
dropped, and SW is interpolated linearly in total variance between the two expiries bracketing the
target. If no pair brackets it (for example late in the month, when every remaining expiry is beyond
30 days) the same formula extrapolates from the two nearest and the row is flagged `extrapolated`.
Its realized variance is measured over the 30 calendar days after the snapshot on the contract the
first valid expiry is written on (`constant_maturity_rv`).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from esvrp.vrp.realized import RV_COLUMNS, rv_table

KINDS = ("diff", "log", "swap_return")
KEYS = ["date", "expiry"]
SW_COLUMNS = ("sw_mid", "sw_bid", "sw_ask")
VRP_COLUMNS = ["vrp_diff", "vrp_log", "vrp_swap_return"]
_KEEP_STRIP = ["underlying", "tau", "snapshot_ts", "n_strikes", "truncation_flag"]
_KEEP_RV = ["n_5m", "n_daily", "start_gap_s", "end_gap_s"]
CM_COLUMNS = ["date", "snapshot_ts", "expiry", "tau", "target_days", "expiry1", "expiry2", "tau1", "tau2",
              "underlying", "extrapolated", *SW_COLUMNS]


def vrp_table(
    strips: pd.DataFrame, rv: pd.DataFrame, kind: Literal["diff", "log", "swap_return"] = "diff",
    sw_col: str = "sw_mid", rv_col: str = "rv_5m",
) -> pd.DataFrame:
    """RV - SW, ln(RV/SW) or RV/SW - 1 (§4.4), one row per chain.

    `strips` (from `strip.strip_table`) and `rv` (from `realized.rv_table`) are joined on date and
    expiry. `sw_col` picks the strip variant (`sw_mid`, `sw_bid`, `sw_ask`) and `rv_col` the realized
    variance (`rv_5m`, `rv_daily` or their `_simple` variants). All three premiums are returned
    (`vrp_diff`, `vrp_log`, `vrp_swap_return`) and `vrp` is the one chosen by `kind`. The log and the
    ratio are NaN unless SW and RV are positive.
    """
    # Decided (§15.6): RV - SW is primary; ln(RV/SW) reported alongside for inference.
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    for name, df, col in (("strips", strips, sw_col), ("rv", rv, rv_col)):
        missing = [c for c in [*KEYS, col] if c not in df]
        if missing:
            raise ValueError(f"{name} is missing columns {missing}")
    left = strips[[*KEYS, sw_col, *(c for c in _KEEP_STRIP if c in strips)]].rename(columns={sw_col: "sw"})
    right = rv[[*KEYS, rv_col, *(c for c in _KEEP_RV if c in rv)]].rename(columns={rv_col: "rv"})
    out = left.merge(right, on=KEYS, how="inner").sort_values(KEYS).reset_index(drop=True)
    sw, r = out["sw"].astype(float), out["rv"].astype(float)
    positive = (sw > 0) & (r > 0)
    out["vrp_diff"] = r - sw
    out["vrp_log"] = np.where(positive, np.log(r.where(positive) / sw.where(positive)), np.nan)
    out["vrp_swap_return"] = (r / sw.where(sw > 0)) - 1.0
    out["vrp"] = out[{"diff": "vrp_diff", "log": "vrp_log", "swap_return": "vrp_swap_return"}[kind]]
    return out


def _interpolate(sw1: pd.Series, sw2: pd.Series, tau1: float, tau2: float, tau: float) -> pd.Series:
    """Carr-Wu Eq. (10): linear in total variance SW * tau, then divided by the target tau."""
    w1, w2 = (tau2 - tau) / (tau2 - tau1), (tau - tau1) / (tau2 - tau1)
    return (sw1 * tau1 * w1 + sw2 * tau2 * w2) / tau


def constant_maturity(
    strips: pd.DataFrame, target_days: int = 30, min_front_days: int = 8,
    sw_cols: tuple[str, ...] = SW_COLUMNS,
) -> pd.DataFrame:
    """Linear interpolation in total variance between the two expiries bracketing target_days.

    One row per date with: `expiry` (snapshot + `target_days`), `tau` (`target_days`/365), the two
    expiries used (`expiry1`, `expiry2`) with their `tau1`, `tau2`, `underlying` of the first valid
    expiry, the interpolated `sw_*` columns and `extrapolated`. Dates with fewer than two valid
    expiries are left out.
    """
    tau_t = target_days / 365
    min_tau = min_front_days / 365
    cols = [c for c in sw_cols if c in strips]
    if not cols:
        raise ValueError(f"strips has none of the columns {sw_cols}")
    pivot = cols[0]
    rows = []
    for date, g in strips.groupby("date", sort=True):
        valid = g[(g["tau"] >= min_tau) & g[pivot].notna()].sort_values("tau")
        if len(valid) < 2:
            continue
        lower, upper = valid[valid["tau"] <= tau_t], valid[valid["tau"] > tau_t]
        extrapolated = not (len(lower) and len(upper))
        if not extrapolated:
            a, b = lower.iloc[-1], upper.iloc[0]
        elif len(upper):  # every valid expiry is beyond the target: use the two nearest
            a, b = valid.iloc[0], valid.iloc[1]
        else:  # every valid expiry is before the target
            a, b = valid.iloc[-2], valid.iloc[-1]
        snap = valid["snapshot_ts"].iloc[0] if "snapshot_ts" in valid else pd.NaT
        row = {
            "date": date, "snapshot_ts": snap,
            "expiry": snap + pd.Timedelta(days=target_days) if pd.notna(snap) else pd.NaT,
            "tau": tau_t, "target_days": target_days, "expiry1": a["expiry"], "expiry2": b["expiry"],
            "tau1": float(a["tau"]), "tau2": float(b["tau"]), "underlying": valid["underlying"].iloc[0],
            "extrapolated": extrapolated,
        }
        for c in cols:
            row[c] = float(_interpolate(pd.Series([a[c]]), pd.Series([b[c]]), a["tau"], b["tau"], tau_t)[0])
        rows.append(row)
    return pd.DataFrame(rows, columns=CM_COLUMNS if not rows else None)


def constant_maturity_rv(cm: pd.DataFrame, bars_5m: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Realized variance over the `target_days` after each snapshot, on the first valid expiry's contract.

    `cm` is the output of `constant_maturity`. Returns the `rv_table` columns, keyed by (date, expiry) so
    it joins to `cm` in `vrp_table`.
    """
    if cm.empty:
        return pd.DataFrame(columns=RV_COLUMNS)
    return rv_table(cm[["date", "expiry", "underlying", "snapshot_ts", "tau"]], bars_5m, daily)


def vrp_descriptives(vrp: pd.DataFrame, columns: tuple[str, ...] = tuple(VRP_COLUMNS)) -> pd.DataFrame:
    """Descriptive statistics of the premiums: one row per column.

    `naive_t` is mean / (std / sqrt(n)) and ignores the overlap between daily observations, so it
    overstates significance. HAC and clustered errors belong to the inference module (roadmap 14).
    """
    rows = {}
    for c in columns:
        x = vrp[c].dropna().astype(float)
        n = len(x)
        rows[c] = {
            "n": n, "mean": x.mean(), "std": x.std(), "min": x.min(), "median": x.median(), "max": x.max(),
            "skew": x.skew(), "excess_kurtosis": x.kurt(),
            "share_negative": float((x < 0).mean()) if n else np.nan,
            "naive_t": x.mean() / (x.std() / np.sqrt(n)) if n > 1 and x.std() > 0 else np.nan,
        }
    return pd.DataFrame(rows).T
