"""Delta-hedged gain engine.

Implements PLANNING.md §4.5.

One option is bought at t_0 and held to expiry T; at each rebalance date t_0 < ... < t_{N-1} the
position is hedged by shorting Delta_n futures. The futures path F_0, ..., F_N runs from entry to
expiry, so F_N is the futures price at expiry and the option pays its intrinsic value there. Gains are
from the long side; a seller earns -Pi.

Hedge delta variants (§4.5):
- `iv_fixed`: Black-76 delta at the entry implied vol `option["iv"]` throughout.
- `iv_daily`: delta at the option's own implied vol on each rebalance date (`hedge_vol`, from the chain).
- `rv`: delta at a trailing realized-vol estimate on each rebalance date (`hedge_vol`, e.g.
  `trailing_vol`), closest to Bakshi-Kapadia's GARCH/rolling-volatility delta.

`build_hedged_gains` runs this for every option and entry date in the chain (§6 stage 6, the daily-start
cross-section used for inference); `run` is the stage entry point.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from esvrp.data.client import DATA_DIR
from esvrp.hedging.costs import ES_MULTIPLIER, apply_costs
from esvrp.pricing.black76 import greeks

log = logging.getLogger(__name__)

SECONDS_PER_YEAR = 365 * 24 * 3600  # ACT/365, as in the chain's tau
TZ = "America/Chicago"
HEDGE_DELTAS = ("iv_daily", "iv_fixed", "rv")
RV_WINDOW = 21  # trading days for the `rv` hedge vol (Bakshi-Kapadia's rolling alternative)


@dataclass
class HedgeResult:
    gain_dollars: float
    gain_over_underlying: float
    gain_over_option: float
    n_hedges: int
    gamma_dt: float = float("nan")  # 1/2 sum Gamma_n F_n^2 dt_n, index points x years
    var_gap: float = float("nan")  # Pi / gamma_dt: gamma-weighted realized minus implied variance


def delta_hedged_gain(
    option: pd.Series,
    underlying_path: pd.Series,
    rate: float,
    hedge_delta: Literal["iv_daily", "iv_fixed", "rv"],
    entry_price: Literal["mid", "ask", "bid"],
    hedge_vol: pd.Series | np.ndarray | None = None,
) -> HedgeResult:
    """Pi = O_T - e^{r tau} O_0 - sum_n Delta_n (F_{n+1} - F_n).

    `option` needs `strike`, `cp`, `tau` (years to expiry at entry), `iv` and the `entry_price` column.
    `underlying_path` is F_0, ..., F_N. With a DatetimeIndex, time to expiry at each rebalance is
    `tau` minus the elapsed time (ACT/365); otherwise the points are taken as equally spaced from entry
    to expiry. `hedge_vol` gives the volatility for delta at each of F_0, ..., F_{N-1} and is required
    for `iv_daily` and `rv` (matched by date when both are dated, else by position); missing values
    carry the last known volatility forward, starting from the entry implied vol.

    Use `entry_price="mid"` together with `apply_costs`, or `"ask"`/`"bid"` without the option
    half-spread there, so the spread is not charged twice.
    """
    F = underlying_path.to_numpy(dtype=float)
    n = len(F) - 1
    if n < 1:
        raise ValueError("underlying_path needs at least an entry and an expiry point")
    K, cp, tau0, iv0 = float(option["strike"]), option["cp"], float(option["tau"]), float(option["iv"])
    O0 = float(option[entry_price])

    taus = _taus_to_expiry(underlying_path.index, tau0, n)
    if isinstance(hedge_vol, pd.Series) and isinstance(underlying_path.index, pd.DatetimeIndex):
        hedge_vol = hedge_vol.reindex(underlying_path.index[:n])
    sigmas = _hedge_sigmas(hedge_delta, hedge_vol, iv0, n)
    pi, gdt = (float(x) for x in _gain_points(F, K, cp, taus, rate, sigmas, O0, tau0))

    return HedgeResult(
        gain_dollars=pi * ES_MULTIPLIER,
        gain_over_underlying=pi / F[0],
        gain_over_option=pi / O0 if O0 > 0 else float("nan"),
        n_hedges=n,
        gamma_dt=gdt,
        var_gap=pi / gdt if gdt > 0 else float("nan"),
    )


def trailing_vol(prices: pd.Series, window: int = 21, periods_per_year: int = 252) -> pd.Series:
    """Annualized rolling sample std of log returns up to and including each date (no look-ahead).

    Bakshi-Kapadia's rolling-volatility alternative to a GARCH hedge vol; feed it as `hedge_vol`
    for `hedge_delta="rv"`.
    """
    log_ret = np.log(prices.astype(float)).diff()
    return log_ret.rolling(window).std() * math.sqrt(periods_per_year)


def _taus_to_expiry(index: pd.Index, tau0: float, n: int) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        elapsed = (index[:n] - index[0]).total_seconds().to_numpy() / SECONDS_PER_YEAR
        taus = tau0 - elapsed
    else:
        taus = tau0 * (1 - np.arange(n) / n)
    if np.any(taus <= 0):
        raise ValueError("underlying_path has rebalance dates at or after expiry")
    return taus


def _hedge_sigmas(hedge_delta: str, hedge_vol: pd.Series | np.ndarray | None, iv0: float,
                  n: int) -> np.ndarray:
    if hedge_delta == "iv_fixed":
        return np.full(n, iv0)
    if hedge_delta not in ("iv_daily", "rv"):
        raise ValueError(f"unknown hedge_delta {hedge_delta!r}")
    if hedge_vol is None:
        raise ValueError(f"hedge_delta={hedge_delta!r} needs hedge_vol")
    vols = np.asarray(hedge_vol, dtype=float)[:n]
    if len(vols) < n:
        raise ValueError("hedge_vol must cover every rebalance date")
    return _ffill(vols, iv0)


def _gain_points(F: np.ndarray, K: float, cp: str, taus: np.ndarray, rate: float, sigmas: np.ndarray,
                 O0: float, tau0: float) -> tuple[np.ndarray, np.ndarray]:
    """Pi in index points and the gamma weight 1/2 sum Gamma_n F_n^2 dt_n for the path F_0..F_N.

    `sigmas` is (N,) or (variants, N), one result per row. Pi / weight is the gamma-weighted average of
    realized minus implied variance (§4.5), annualized like RV - SW.
    """
    n = len(F) - 1
    g = greeks(F[:n], K, taus, rate, sigmas, cp)
    deltas, gammas = np.asarray(g["delta"]), np.asarray(g["gamma"])
    dt = taus - np.append(taus[1:], 0.0)
    payoff = max(F[-1] - K, 0.0) if cp == "c" else max(K - F[-1], 0.0)
    pi = payoff - math.exp(rate * tau0) * O0 - (deltas * np.diff(F)).sum(axis=-1)
    return pi, 0.5 * (gammas * F[:n] ** 2 * dt).sum(axis=-1)


def _ffill(vols: np.ndarray, first: float) -> np.ndarray:
    """Carry the last positive vol forward over missing or non-positive values, starting from `first`."""
    ok = np.isfinite(vols) & (vols > 0)
    if ok.all():
        return vols
    last = np.maximum.accumulate(np.where(ok, np.arange(len(vols)), -1))
    return np.where(last >= 0, vols[np.maximum(last, 0)], first)


ENTRY_COLUMNS = ["instrument_id", "raw_symbol", "underlying", "date", "snapshot_ts", "expiry", "cp",
                 "strike", "moneyness", "tau", "r", "F", "F_fut", "iv", "delta", "vega", "bid", "ask", "mid"]
GAIN_COLUMNS = [*ENTRY_COLUMNS, "hedge_delta", "n_hedges", "F_T", "gain", "gain_over_F",
                "gain_over_option", "gamma_dt", "var_gap", "halfspread", "cost"]


def build_hedged_gains(
    chain: pd.DataFrame,
    daily: pd.DataFrame,
    moneyness_max: float = 0.10,
    days_range: tuple[float, float] = (14, 60),
    hedge_deltas: tuple[str, ...] = HEDGE_DELTAS,
    option_halfspread: float | str = "from_data",
    slippage_ticks: float = 0.5,
    fee_per_contract: float = 1.50,
) -> tuple[pd.DataFrame, pd.Series]:
    """Hedged gain of every eligible option from each entry date to expiry, once per hedge delta.

    Entries: options with |moneyness| <= `moneyness_max`, `days_range` days to expiry, and an entry mid
    and implied vol. Each is bought at the chain's mid and hedged daily at the 15:00 CT snapshot with its
    own futures contract (`F_fut`; the parity `F` is only the pricing forward behind `iv`). The option
    pays intrinsic value at the futures settlement on its expiry date, taken as the 15:00 CT price EW
    options are exercised against. Rebalance dates are the snapshots on which the contract has a futures
    price, so a missing day just lengthens one hedge interval.

    `gain` is the long-side gross gain in dollars at mid; `cost` (dollars, positive) is the half-spread
    (`(ask - bid) / 2` at entry for `"from_data"`), slippage and fees from `apply_costs`. Net gains are
    `gain - cost` for a buyer and `-gain - cost` for a seller.

    Returns the gains (one row per option, entry date and hedge delta) and the number of entries
    dropped for each reason.
    """
    for hd in hedge_deltas:
        if hd not in HEDGE_DELTAS:
            raise ValueError(f"unknown hedge_delta {hd!r}")
    has_mid = chain["mid"].notna()
    eligible = (chain["moneyness"].abs() <= moneyness_max) & (chain["tau"] * 365).between(*days_range)
    entries = chain[eligible & has_mid & chain["iv"].notna()].copy()
    dropped = {"one_sided": int((eligible & ~has_mid).sum()),
               "no_iv": int((eligible & has_mid & chain["iv"].isna()).sum())}

    # The option pays off at its contract's settlement on the expiry date.
    # TODO(docs/data.md item 9): verify EW options are exercised against the 15:00 CT settlement.
    entries["expiry_date"] = pd.to_datetime(entries["expiry"], utc=True).dt.tz_convert(TZ).dt.date
    after = entries["expiry_date"] > chain["date"].max()
    settle = daily.dropna(subset=["settle"]).set_index(["underlying", "date"])["settle"]
    keys = pd.MultiIndex.from_arrays([entries["underlying"], entries["expiry_date"]])
    entries["F_T"] = settle.reindex(keys).to_numpy()
    dropped["expires_after_sample"] = int(after.sum())
    dropped["no_expiry_settle"] = int((~after & entries["F_T"].isna()).sum())
    entries = entries[~after & entries["F_T"].notna()]

    rv = _trailing_rv(daily)
    used = chain[chain["underlying"].isin(entries["underlying"].unique())]
    grids = {u: _Grid.build(g, rv.get(u)) for u, g in used.groupby("underlying")}
    fixed_spread = None if option_halfspread == "from_data" else float(option_halfspread)

    rows = []
    for opt in entries.itertuples(index=False):
        grid = grids[opt.underlying]
        entry_ns, expiry_ns = pd.Timestamp(opt.snapshot_ts).value, pd.Timestamp(opt.expiry).value
        lo, hi = np.searchsorted(grid.ts, entry_ns), np.searchsorted(grid.ts, expiry_ns)
        taus = opt.tau - (grid.ts[lo:hi] - entry_ns) / 1e9 / SECONDS_PER_YEAR
        F = np.append(grid.F[lo:hi], opt.F_T)
        sigmas = np.vstack([grid.hedge_vols(hd, opt.instrument_id, lo, hi, opt.iv) for hd in hedge_deltas])
        pis, gdts = _gain_points(F, opt.strike, opt.cp, taus, opt.r, sigmas, opt.mid, opt.tau)

        n = hi - lo
        halfspread = (opt.ask - opt.bid) / 2 if fixed_spread is None else fixed_spread
        cost = -apply_costs(0.0, halfspread, n, slippage_ticks, fee_per_contract)
        entry = [getattr(opt, c) for c in ENTRY_COLUMNS]
        for hd, pi, gdt in zip(hedge_deltas, pis, gdts, strict=True):
            rows.append([*entry, hd, n, opt.F_T, pi * ES_MULTIPLIER, pi / F[0], pi / opt.mid,
                         gdt, pi / gdt if gdt > 0 else np.nan, halfspread, cost])
    return pd.DataFrame(rows, columns=GAIN_COLUMNS), pd.Series(dropped, name="dropped")


@dataclass
class _Grid:
    """One futures contract on its snapshot grid: times, `F_fut`, trailing rv and every option's iv."""

    ts: np.ndarray  # snapshot times with a futures price, int64 ns UTC
    F: np.ndarray
    rv: np.ndarray
    iv: np.ndarray  # (snapshots, options)
    col: dict[Any, int]  # instrument_id -> column of `iv`

    @classmethod
    def build(cls, chain: pd.DataFrame, rv: pd.Series | None) -> _Grid:
        """`chain` holds one contract's rows; `rv` is its trailing vol by trade date (`_trailing_rv`)."""
        c = chain.dropna(subset=["F_fut"])
        snaps = c.groupby("snapshot_ts").agg(F=("F_fut", "first"), date=("date", "first")).sort_index()
        iv = c.pivot_table(index="snapshot_ts", columns="instrument_id", values="iv", aggfunc="first",
                           dropna=False).reindex(snaps.index)
        rv = pd.Series(dtype=float) if rv is None else rv
        return cls(ts=snaps.index.as_unit("ns").asi8, F=snaps["F"].to_numpy(),
                   rv=rv.reindex(snaps["date"]).to_numpy(dtype=float),
                   iv=iv.to_numpy(dtype=float), col={k: i for i, k in enumerate(iv.columns)})

    def hedge_vols(self, hedge_delta: str, instrument_id: Any, lo: int, hi: int, iv0: float) -> np.ndarray:
        if hedge_delta == "iv_fixed":
            return np.full(hi - lo, iv0)
        vols = self.iv[lo:hi, self.col[instrument_id]] if hedge_delta == "iv_daily" else self.rv[lo:hi]
        return _ffill(vols, iv0)


def _trailing_rv(daily: pd.DataFrame) -> dict[str, pd.Series]:
    """Per contract: trailing settlement-to-settlement vol through the previous trade date, by date.

    Lagged one day because a day's settlement is published after the 15:00 CT snapshot it would hedge.
    """
    d = daily.dropna(subset=["settle"]).sort_values("date")
    return {u: trailing_vol(g.set_index("date")["settle"], RV_WINDOW).shift(1)
            for u, g in d.groupby("underlying")}


def run(cfg: Any) -> pd.DataFrame:
    """Stage 6 entry point: `data/hedged_gains.parquet` from the chain and the daily futures table."""
    chain = pd.read_parquet(DATA_DIR / "chain.parquet")
    daily = pd.read_parquet(DATA_DIR / "underlying_daily.parquet")
    buckets = cfg.hedging.maturity_buckets_days
    gains, dropped = build_hedged_gains(
        chain, daily, moneyness_max=cfg.hedging.moneyness_max, days_range=(buckets[0], buckets[-1]),
        option_halfspread=cfg.costs.option_halfspread, slippage_ticks=cfg.costs.hedge_slippage_ticks,
        fee_per_contract=cfg.costs.fee_per_contract_usd)
    gains.to_parquet(DATA_DIR / "hedged_gains.parquet")
    base = gains[gains["hedge_delta"] == cfg.hedging.hedge_delta]
    log.info("Hedged gains: %d entries x %d hedge deltas; dropped %s; %s: mean long gain %.1f, mean cost "
             "%.1f dollars", len(base), gains["hedge_delta"].nunique(), dropped.to_dict(),
             cfg.hedging.hedge_delta, base["gain"].mean(), base["cost"].mean())
    return gains
