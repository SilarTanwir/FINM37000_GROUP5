# Stage 5: vrp

See `docs/PLANNING.md` §6 for inputs, outputs and the theory section (§4.2–4.4). Stage 5 combines the
variance swap rate (roadmap 9), realized variance (roadmap 10) and the premium (roadmap 11).

Status: roadmaps 9 (strip), 10 (realized variance) and 11 (VRP table, constant maturity, descriptives,
plots) are implemented, wired into `python -m esvrp.stages 5` / `doit vrp`, and tested end to end on
simulated data. Not yet run on real data (see the TODO at the end). No real
data has been run through it yet.

## Variance-swap strip (`src/esvrp/vrp/strip.py`, roadmap 9)

Input: one chain of `data/chain.parquet` (one expiry on one day, §4.1) with its forward `F`, rate `r` and
`tau`. Output: one annualized variance number, the synthetic variance swap rate `SW` (0.04 is a 20% vol).

    SW = (2/tau) e^{r tau} sum_i (dK_i / K_i^2) Q(K_i) - (1/tau) (F/K0 - 1)^2

- `K0` is the highest quoted strike at or below `F` (the parity forward of stage 4).
- `Q(K_i)` is the out-of-the-money price: put below `K0`, call above, average of put and call at `K0`.
  If only one side is quoted at `K0` it is used alone and `k0_one_sided` is set.
- `dK_i` is half the distance between neighbouring *used* strikes (one-sided at the ends), so a strike
  dropped in stage 2 for a zero bid simply widens the neighbours' `dK`.
- `variance_swap_rate(chain_slice, F, r, tau, price_col)` returns a `StripResult` with `sw`, `k0`,
  `n_strikes`, `strike_min`, `strike_max`, `truncation_flag`, per-strike `contributions`, `max_strike_gap`
  and the subtracted `correction`.
- `price_col` is `mid`, `bid` or `ask`. Quotes with a missing or non-positive price are ignored, so a
  one-sided quote enters the `bid` strip but not `mid` or `ask`. Carr-Wu report all three (§4.3).
- **Implied-vol bounds:** quotes whose implied vol is outside [`strip.min_iv`, `strip.max_iv`] (1% to 100%, as
  in Bakshi-Kapadia) are dropped before the strip is built, in every variant. A single junk quote far from
  the money can otherwise dominate the sum: a user-defined `EWQ6 C10100` call quoted at about 508% vol
  contributed 1.00 to a swap rate of about 0.03. The chain's `iv` (mid) is used where it exists; quotes
  without one (one-sided quotes) get their implied vol from the price used. `n_dropped_iv` counts them;
  `max_iv=None` turns the cap off.
- **No quotes near the money:** `k0_gap_pct = (F - K0) / F` is recorded. If it exceeds
  `strip.max_k0_gap_pct` (1%), the swap rate is left empty (NaN) and `k0_gap_flag` is set. Otherwise a hole
  around the money (stage 2 drops strikes with no bid and no ask) pushes K0 far below F, and the correction
  `(F/K0 - 1)^2 / tau` outweighs the sum: on real chains more than 60 days out this gave negative swap
  rates (7 of 452 chains in July-September 2026; for example 22 July, December expiry, `sw_mid` -0.113 with
  correction 0.140). `truncation_flag` does not catch it, because the strikes still span far on both
  sides. The premium of a flagged chain is NaN, and `constant_maturity` skips it. `None` turns the guard
  off. Not done: a second rule on the number of two-sided quotes within a few strikes of F (issue #42).
- `truncation_flag` is set when the strikes do not span `F(1 - m)` to `F(1 + m)` with
  `m = cleaning.min_range_pct` (10%). Truncation biases `SW` downward.
- Invalid `F` or `tau`, or no usable OTM option, gives NaN with `n_strikes == 0`.
- `strip_table(chain)` runs every chain and returns one row per (date, expiry) with `sw_mid`, `sw_bid`,
  `sw_ask` and the mid-strip diagnostics. It will be written to `data/strip.parquet` once the stage is
  wired.

Tests (`tests/test_strip_flat_vol.py`, no data): `SW` ~ sigma^2 on a dense flat-vol grid, error shrinks
with a finer grid, a narrow range biases `SW` down and sets the flag, a hand-computed three-strike case,
the correction term, discounting, bid < mid < ask, one-sided quotes, gaps in strikes and bad inputs.

## Sources

- Carr and Wu (2009), Proposition 1, Eq. (5): the continuous result, `E^Q[RV] = (2/(T-t)) integral of
  Theta(K)/(B K^2) dK + jump error`, with `e^{r tau} = 1/B`.
- Cboe VIX methodology (`docs/references/Cboe VIX Mathematics Methodology.pdf`, section "Calculating
  Volatility", document pp. 7-8): the discrete sum, `K0`, `dK`, `Q(K0)` and the `(F/K0 - 1)^2` term.
  Checked against the document: formula, `dK` rule, `K0` as the strike at or immediately below `F`,
  `Q(K0)` as the put/call average, `F = K + e^{RT}(C - P)` at the smallest `|C - P|` (stage 4).
- Carr and Wu do not use the discrete sum; their grid method is the `iv_grid` robustness variant.

Differences from the Cboe rules:
- Cboe stops outward after two consecutive zero-bid or zero-ask strikes. Stage 2 already drops zero-bid
  rows, so that cutoff cannot be applied here; strikes beyond a gap are kept with a wider `dK`
  (`max_strike_gap`).
- Cboe requires both a put and a call at `K0`. Here one side is used and `k0_one_sided` is set.
- `tau` is ACT/365 from the snapshot, as in Cboe's minute-based year.

Not yet implemented: the Carr-Wu `iv_grid` variant (§4.3, `strip.method`), jump and stale-quote
diagnostics beyond `max_strike_gap`, and running on real chains.

## Realized variance (`src/esvrp/vrp/realized.py`, roadmap 10)

For each chain (one expiry on one day), `RV = (1/tau) * sum(r_i^2)` over the window from the snapshot
(15:00 CT) to the option's expiry, on the futures contract the option is written on (`underlying`).
`tau` is the chain's own ACT/365 time to expiry, the same `tau` the strip uses, so `RV` and `SW` are
annualized on the same calendar-year basis (§4.2).

- **5-minute:** closes of the traded bars of stage 3. The base is the last bar ending at or before the
  snapshot; the last return ends at the last bar ending at or before the expiry (the contract keeps
  trading after the option expires, so the window is cut there).
- **Daily:** settlement to settlement. Base is the last settlement on or before the snapshot's trade
  date, the end is the last settlement on or before the expiry's trade date. Days without a settlement
  (shortened holiday sessions) are skipped.
- **Gaps:** each return runs from one bar's close to the next, so the move across the maintenance halt,
  a weekend or a holiday is one return spanning the gap (§15.3). The test checks that the returns
  telescope to the log price move, which fails if any gap is dropped or counted twice.
- **Variants:** `kind="log"` (baseline) or `"simple"` (Carr-Wu use simple returns).
- `realized_variance(returns, tau)`, `returns_from_bars(bars, freq, session, start, end, underlying,
  kind)` and `rv_table(chains, bars_5m, daily)`. `rv_table` returns one row per chain: `rv_5m`,
  `rv_daily`, their `_simple` variants, `n_5m`, `n_daily`, and `start_gap_s` / `end_gap_s` (seconds from
  the snapshot to the base bar and from the last bar to the expiry), which flag thin windows or an
  expiry that does not match the bars.
- `session` is validated here; the full-Globex / RTH filter is applied when the bars are built (stage 3).
- **Windows the bars do not cover have no RV.** If the first or last bar misses its end of the window by
  more than `realized.max_gap_days` (default 2: a weekend or a holiday), the RV columns of that frequency are
  NaN and `complete_5m` / `complete_daily` is False. Summing the returns that exist and dividing by the full
  `tau` would understate RV and bias the premium toward H1. The gaps stay in `start_gap_s`, `end_gap_s`
  (5-minute) and `start_gap_days`, `end_gap_days` (daily). This happens when the data end before an
  expiry, when a contract stops trading before a constant 30-day window ends (the third Friday of its
  month), or when stage 3 stopped pulling a contract at its last option expiry (see the 30-day tail in
  stage 3). The premium of such a chain is NaN and the descriptives leave it out.

Open, to settle on the real data (issue #32): the actual expiry time of EW options, and thin underlying
contracts (sparse or stale 5-minute bars).

Tests (`tests/test_realized_variance.py`, no data): convergence to sigma^2 on simulated GBM at daily and
5-minute sampling, log vs simple, the gap case (halt, weekend, Sunday reopen) through stage 3's
`build_bars`, the snapshot-to-expiry window, daily skipping of days without a settlement, `rv_table`
against a manual window, gap flags and missing contracts.

## VRP series and constant maturity (`src/esvrp/vrp/premium.py`, roadmap 11)

- `vrp_table(strips, rv, kind, sw_col, rv_col)` joins `strip_table` and `rv_table` on (date, expiry) and
  returns `sw`, `rv`, `vrp_diff` (`RV - SW`, primary, §15.6), `vrp_log` (`ln(RV/SW)`, reported alongside
  for inference), `vrp_swap_return` (`RV/SW - 1`) and `vrp` (the one chosen by `kind`). The log and the
  ratio are NaN unless SW and RV are positive. `sw_col` selects mid, bid or ask; `rv_col` selects 5-minute
  or daily, log or simple.
- `constant_maturity(strips, 30)` follows Carr-Wu Eq. (10): drop expiries under `min_front_days` (8),
  take the pair that brackets the target and interpolate linearly in total variance
  (`SW_T = [SW1 tau1 (tau2 - tau) + SW2 tau2 (tau - tau1)] / ((tau2 - tau1) tau)`). If no pair brackets the
  target (for example every valid expiry is beyond 30 days, late in the month) the same formula
  extrapolates from the two nearest expiries and `extrapolated` is set. A flat swap rate stays flat.
  The swap rate is left empty (`sw_*` NaN, reason in `cm_reason`) when the nearest valid expiry is more
  than `strip.max_cm_extrapolation_days` (60) beyond the snapshot (`extrapolation_too_far`) or when the
  extrapolated total variance is not positive (`non_positive_variance`); otherwise `cm_reason` is `ok`
  (#55). The row is kept so the lost dates can be counted, and its premium is NaN.
- `constant_maturity_rv(cm, bars_5m, daily)`: RV over the 30 calendar days after the snapshot, on the
  contract the first valid expiry is written on (decision; the two expiries can sit on different
  contracts). Keyed by (date, expiry = snapshot + 30 days) so `vrp_table(cm, rv)` joins it.
- `vrp_descriptives(vrp)`: n, mean, std, min, median, max, skew, excess kurtosis, share negative and a
  **naive** t-statistic. It ignores the overlap between daily observations and overstates significance;
  HAC and clustered errors belong to the inference module (roadmap 14).
- Plots (`src/esvrp/reporting/figures.py`): `sw_vs_rv_figure` (SW and RV over time, 45-degree scatter),
  `vrp_distribution_figure` (histogram and rolling mean), `rv_comparison_figure` (daily vs 5-minute RV).
- `strip_table` now also carries `snapshot_ts`, which `constant_maturity` needs.

Tests (`tests/test_premium.py`, no data): hand-computed premiums for all three kinds, variant selection,
NaN handling, Eq. (10) by hand, flat variance preserved under interpolation and extrapolation, the 8-day
rule, bracketing preference, skipped dates, constant-maturity RV against a manual window, descriptives and
the figures (including empty input).

## H1 test with Newey-West errors (`h1_table`, issue #52)

`vrp_descriptives` reports a naive t, which overstates significance because daily-start 30-day windows
overlap (§4.6). `h1_table(vrp, vrp_cm, hac_lags)` tests H1 (mean premium < 0) for `vrp_diff` and
`vrp_log` on two samples:

| Sample | Observations | HAC lag |
|---|---|---|
| `constant 30-day (daily, overlapping)` | `vrp_cm`, one per date | 20 (horizon of 21 trading days minus one, `inference.hac_lags: auto`), plus `t_hac_30` with Carr-Wu's 30 |
| `matched, monthly (non-overlapping)` | One chain per expiry, the first date in its month (`monthly_subsample`, the strategy's entry rule) | 6 (Carr-Wu) |

Columns: `n`, `mean`, `se_naive`, `t_naive`, `hac_lags`, `se_hac`, `t_hac`, `t_hac_30` and, for the log
premium, `sharpe_short`: Carr-Wu's Sharpe ratio for shorting variance, the mean of -ln(RV/SW) over its
Newey-West standard deviation, annualized by sqrt(365/30) (daily) or sqrt(12) (monthly). A negative mean
with `t_hac` below about -2 supports H1; the naive t is kept to show how much the overlap inflates it.

Tests (`tests/test_premium.py`): on simulated premia that share 20 of 21 days, the HAC |t| is less than
half the naive |t|; the monthly subsample has one row per expiry with lag 6; the Sharpe ratio by hand; a
configured lag; empty inputs.

## Running stage 5 (`src/esvrp/vrp/pipeline.py`)

`python -m esvrp.stages 5` (or `doit vrp`) needs `data/chain.parquet` (stage 4) and
`data/underlying_5m.parquet` / `data/underlying_daily.parquet` (stage 3). No data is pulled. Outputs:

| File | Content |
|---|---|
| `data/strip.parquet` | `strip_table`: `sw_mid`, `sw_bid`, `sw_ask` and strip diagnostics, one row per chain |
| `data/rv.parquet` | `rv_table`: 5-minute and daily RV, log and simple, return counts and gap diagnostics |
| `data/vrp.parquet` | `vrp_table`: `sw`, `rv`, `vrp_diff` (primary), `vrp_log`, `vrp_swap_return`, at each expiry's own maturity |
| `data/strip_cm.parquet`, `rv_cm.parquet`, `vrp_cm.parquet` | The constant 30-day series (Carr-Wu comparable), with `extrapolated` |
| `reports/tables/vrp_descriptives.csv`, `vrp_cm_descriptives.csv` | `vrp_descriptives` (naive t only) |
| `reports/tables/h1.csv` | `h1_table`: the H1 test with naive and Newey-West t-statistics |
| `reports/figures/sw_vs_rv.png`, `vrp_distribution.png`, `rv_comparison.png` | Plots 1-3 for every chain |
| `reports/figures/cm_sw_vs_rv.png`, `cm_vrp_distribution.png` | The same for the constant-maturity series |

Config used: `strip.max_k0_gap_pct` (the K0 guard above), `strip.min_iv` / `strip.max_iv`, `realized.max_gap_days`, `strip.price_col` (which `sw_*` column is the premium's SW; default mid),
`strip.min_front_days` (8), `cleaning.min_range_pct` (truncation flag) and `realized.frequency` (`rv_5m`
is the primary RV if `5m` is listed, otherwise `rv_daily`). `strip.method: iv_grid` raises
`NotImplementedError` until issue #31 is done. The target maturity is fixed at 30 days (`TARGET_DAYS`).
`run(cfg, data_dir, report_dir)` takes its folders as arguments, which the tests use.

The constant-maturity RV window is 30 calendar days from the snapshot. `end_gap_s` in `rv_cm.parquet`
shows how far the last bar is from the end of that window; it is large when the bars stop early.

Tests: `tests/test_stage5.py` runs the whole stage on simulated chains and bars (all files written,
one row per chain, flat-vol swap rate, extrapolation flag, no constant maturity when fewer than two valid
expiries, unimplemented method, empty chain, the `stages.py` and `dodo.py` wiring).

## TODO: run stage 5 on real data (issue #35, Part B)

- Run stages 1-4 and then stage 5 on the full sample.
- Sanity-check the strip (truncation flags, `n_strikes`, `max_strike_gap`, bid < mid < ask, levels
  against VIX) and RV (`start_gap_s`, `end_gap_s`, bar counts, 5-minute vs daily agreement).
- Review the real VRP series for H1, remembering that t-statistics are naive until roadmap 14.
- Settle the open points in issue #32 (EW expiry time, thin underlying contracts) and compare with the
  `iv_grid` variant (#31).
