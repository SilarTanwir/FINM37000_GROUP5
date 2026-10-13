# Stage 5: vrp

See `docs/PLANNING.md` §6 for inputs, outputs and the theory section (§4.2–4.4). Stage 5 combines the
variance swap rate (roadmap 9), realized variance (roadmap 10) and the premium (roadmap 11).

Status: roadmap 9 (strip) implemented and tested on simulated chains; roadmaps 10 and 11 pending. No real
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

TODO: realized variance (roadmap 10) and the VRP table, plots and descriptives (roadmap 11).
