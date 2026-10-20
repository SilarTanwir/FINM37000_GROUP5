# Stage 6: hedging

See `docs/PLANNING.md` §6 for inputs, outputs and the theory section (§4.5). Code:
`src/esvrp/hedging/simulator.py` (hedged gains and the stage driver), `src/esvrp/hedging/costs.py`
(costs). Run with `doit hedging` or `python -m esvrp.stages 6` (needs `data/chain.parquet` from stage 4
and `data/underlying_daily.parquet` from stage 3). No data is pulled.

## Hedged gain of one option

`delta_hedged_gain` buys one option at entry, holds it to expiry and shorts `Delta_n` futures at each
rebalance date:

`Pi = O_T - e^{r tau} O_0 - sum_n Delta_n (F_{n+1} - F_n)`

- `underlying_path` is the futures price from entry (`F_0`) to expiry (`F_N`). Rebalance dates are
  `F_0, ..., F_{N-1}`; `O_T` is intrinsic value at `F_N`. A seller earns `-Pi`.
- With a dated path, time to expiry at each rebalance is the entry `tau` minus elapsed ACT/365 time,
  matching the chain's `tau`. An undated path is taken as equally spaced from entry to expiry.
- `n_hedges` is `N`, the number of rebalance dates, which is what `apply_costs` charges slippage on.

## Hedge delta variants

| `hedge_delta` | Volatility for Black-76 delta | In the stage driver |
|---|---|---|
| `iv_fixed` | Entry implied vol | the entry quote's `iv` |
| `iv_daily` (baseline) | The option's own implied vol each day | the option's `iv` in each day's chain |
| `rv` | Trailing realized vol (Bakshi-Kapadia's rolling alternative to GARCH) | 21-day settlement-to-settlement vol of the option's own contract, through the previous trade date |

Missing values (a day with no two-sided quote, the first weeks of a contract's history) carry the last
known vol forward, starting from the entry implied vol. The `rv` vol is lagged a day because a day's
settlement is published after the 15:00 CT snapshot it would hedge.

## Stage driver: `build_hedged_gains`

The daily-start cross-section used for inference (§6): every option on every snapshot date that passes
the entry rules, hedged to expiry once per hedge delta.

**Entries.** `|moneyness| <= hedging.moneyness_max` (10%), 14 to 60 days to expiry (the first and last
`hedging.maturity_buckets_days`), and a two-sided quote with an implied vol. Bought at the chain's `mid`.

**Futures path.** `F_fut`, the option's own futures contract at the 15:00 CT snapshot, on every snapshot
date from entry until before expiry. Not the chain's `F`: that is the parity forward used to compute `iv`,
while the hedge trades the futures contract itself. The two differ by about 1 point at the median
(`docs/stages/04_chain.md`). A day on which the contract has no futures price is skipped, which just
lengthens one hedge interval.

**Expiry.** EW options expire at 15:00 CT on the last business day of the month. The option pays
intrinsic value at its futures contract's **settlement on the expiry date** (stage 3 `settle`), taken as
the price EW options are exercised against (open item 9 in `docs/data.md`). Expiring options are not in
that day's chain (stage 2 excludes options at or past expiry), so this is the only price for the last
step.

**Drops** (returned with the gains and logged): `one_sided` and `no_iv` (no entry mid or implied vol),
`expires_after_sample` (expiry after the last chain date) and `no_expiry_settle`.

### Output: `data/hedged_gains.parquet`

One row per option, entry date and hedge delta. Entry columns from the chain (`instrument_id`,
`raw_symbol`, `underlying`, `date`, `snapshot_ts`, `expiry`, `cp`, `strike`, `moneyness`, `tau`, `r`, `F`,
`F_fut`, `iv`, `delta`, `vega`, `bid`, `ask`, `mid`), plus:

| Column | Meaning |
|---|---|
| `hedge_delta` | `iv_daily`, `iv_fixed` or `rv`; `hedging.hedge_delta` picks the baseline |
| `n_hedges` | Rebalance dates from entry to expiry |
| `F_T` | Settlement of the futures contract on the expiry date |
| `gain` | `Pi` in dollars ($50 per point), long side, entry at mid |
| `gain_over_F`, `gain_over_option` | `Pi / F_0` and `Pi / O_0` (Bakshi-Kapadia) |
| `gamma_dt` | `1/2 sum_n Gamma_n F_n^2 dt_n` (index points × years), with gamma at the hedge vol |
| `var_gap` | `Pi / gamma_dt`: gamma-weighted realized minus implied variance, annualized. For one option it is biased up by about 0.0015 at daily hedging; sum gains and weights over options first (stage 7, H3) |
| `halfspread` | Entry half-spread in index points: `(ask - bid) / 2` for `costs.option_halfspread: from_data`, else the configured number |
| `cost` | Dollars, positive: half-spread, slippage and fees from `apply_costs` |

Net gains are `gain - cost` for a buyer and `-gain - cost` for a seller; costs do not depend on the side.
Entry `vega` is kept for the per-unit-vega returns of the strategy (§4.7).

**Speed.** About 0.85 ms per entry for all three hedge deltas (one Black-76 call per entry), so a few
minutes for the full sample.

## Costs

`apply_costs` charges the half-spread once (held to expiry, no exit spread), 0.5 tick ($6.25) of
slippage per rebalance and $1.50 per contract on the option and on each hedge trade. When calling
`delta_hedged_gain` directly, use `entry_price="mid"` with `apply_costs`, or `"ask"`/`"bid"` without
the option half-spread there, so the spread is not charged twice.

## Checks

- `test_hedge_zero_premium`: on simulated futures paths, the mean gain is zero when realized vol equals
  implied vol and positive when it is higher. At 18% implied and 30% realized, a 30-day ATM call gains
  about $3,490 on average, close to the vega approximation of $3,420.
- `test_hedge_identity`: gains co-move with the gamma-weighted realized-minus-implied variance.
- `test_hedge_simulator`: exact values on short paths, calendar-time `tau`, put-call parity of hedged
  gains, and the three hedge-delta variants.
- `test_hedged_gains`: on a synthetic month of chains, the driver matches `delta_hedged_gain` on the
  `F_fut` path with the expiry settlement; entry filters and drop counts; `iv_daily` follows the option's
  own implied vol; the `rv` vol never uses that day's settlement; the stage writes the parquet file.

Not yet run on real data: `data/chain.parquet` is not built in this checkout.

## Known limitations

- Slippage assumes one futures contract per rebalance, whatever the change in delta.
- The `rv` hedge vol has no history for the first month of the sample or of a contract's settlements,
  and falls back to the entry implied vol there.
