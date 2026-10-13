# Stage 6: hedging

See `docs/PLANNING.md` §6 for inputs, outputs and the theory section (§4.5). Code:
`src/esvrp/hedging/simulator.py` (hedged gains), `src/esvrp/hedging/costs.py` (costs, used in stage 7).

The stage driver (`python -m esvrp.stages 6`, reading `data/chain.parquet` and
`data/underlying_daily.parquet`) is not written yet; it waits for roadmap 8 to merge.

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

| `hedge_delta` | Volatility for Black-76 delta | `hedge_vol` |
|---|---|---|
| `iv_fixed` | Entry implied vol `option["iv"]` | not used |
| `iv_daily` (baseline) | The option's own implied vol each day | the option's `iv` column from the chain |
| `rv` | Trailing realized vol | `trailing_vol(F_daily, 21)`, a 21-day rolling std (Bakshi-Kapadia's alternative to GARCH) |

`hedge_vol` is matched to the path by date. Missing values (e.g. a day with no two-sided quote) carry
the last known vol forward, starting from the entry implied vol.

## Normalizations

`HedgeResult` reports `Pi` in dollars ($50 per point), `Pi / F_0` and `Pi / O_0` (Bakshi-Kapadia).

## Costs

`entry_price` picks `mid`, `ask` or `bid` for `O_0`. Use `mid` together with `apply_costs`, or `ask`/`bid`
without the option half-spread there, so the spread is not charged twice. `apply_costs` charges the
half-spread once (held to expiry, no exit spread), 0.5 tick ($6.25) of slippage per rebalance and
$1.50 per contract on the option and on each hedge trade.

## Checks

- `test_hedge_zero_premium`: on simulated futures paths, the mean gain is zero when realized vol equals
  implied vol and positive when it is higher. At 18% implied and 30% realized, a 30-day ATM call gains
  about $3,490 on average, close to the vega approximation of $3,420.
- `test_hedge_identity`: gains co-move with the gamma-weighted realized-minus-implied variance.
- `test_hedge_simulator`: exact values on short paths, calendar-time `tau`, put-call parity of hedged
  gains, and the three hedge-delta variants.

## Known limitations

- Slippage assumes one futures contract per rebalance, whatever the change in delta.
