# Stage 5: vrp

See `docs/PLANNING.md` §6 for inputs, outputs and the theory section (§4.2–4.4). Stage 5 combines the
variance swap rate (roadmap 9), realized variance (roadmap 10) and the premium (roadmap 11).

Status: roadmap 9 (strip) and roadmap 10 (realized variance) implemented and tested on simulated data; roadmap 11 pending. No real
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

Open, to settle on the real data (issue #32): the actual expiry time of EW options, and thin underlying
contracts (sparse or stale 5-minute bars).

Tests (`tests/test_realized_variance.py`, no data): convergence to sigma^2 on simulated GBM at daily and
5-minute sampling, log vs simple, the gap case (halt, weekend, Sunday reopen) through stage 3's
`build_bars`, the snapshot-to-expiry window, daily skipping of days without a settlement, `rv_table`
against a manual window, gap flags and missing contracts.

TODO: the VRP table, plots and descriptives (roadmap 11).
