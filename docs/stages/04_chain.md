# Stage 4: chain

See `docs/PLANNING.md` §6 for inputs, outputs and the theory section (§4.1). Code:
`src/esvrp/pricing/chain.py` (assembly), `src/esvrp/pricing/forward.py` (parity forward),
`src/esvrp/data/rates.py` (rate). Run with `doit chain` or `python -m esvrp.stages 4` (needs
`data/quotes.parquet` from stage 2 and `data/underlying_daily.parquet` from stage 3). No data is pulled.

A **chain** is one expiry on one day. The stage keeps every cleaned quote from stage 2 and adds the
columns the strip (roadmap 9) and the hedge simulator (roadmap 12) read.

## Output: `data/chain.parquet`

All stage 2 columns (`date`, `snapshot_ts`, `instrument_id`, `raw_symbol`, `cp`, `strike`, `expiry`,
`underlying`, `user_defined`, `bid`, `ask`, `bid_sz`, `ask_sz`, `quote_ts`, `quote_age_s`, `one_sided`),
except that stage 2's `F` is renamed `F_fut`, plus:

| Column | Meaning |
|---|---|
| `tau` | Years from the snapshot to expiry, ACT/365 |
| `r` | Continuously compounded rate for the date (`pricing.rate_source`; only `constant` is implemented) |
| `mid` | `(bid + ask) / 2`. Empty for `one_sided` quotes, which keep their `bid` |
| `F_fut` | Futures mid of the option's own underlying at the snapshot (from stage 2) |
| `F_parity` | Forward implied by put-call parity for the chain |
| `F` | **The forward to use for pricing**: `F_parity`, or `F_fut` where there is no parity forward |
| `forward_gap` | `F_parity - F_fut`, index points |
| `settle` | Settlement price of the underlying that day (from stage 3), for reference |
| `moneyness` | `ln(K / F)` (§4.5) |
| `iv` | Black-76 implied volatility of `mid` at `F`, `tau`, `r` |
| `delta`, `gamma`, `vega` | Black-76 Greeks with respect to `F`, at the option's own `iv` |

`F`, `F_fut`, `F_parity`, `forward_gap`, `tau` and `r` are the same on every row of a chain. No row is
dropped: screens such as the moneyness and implied-vol bounds of §3.2 belong to the analysis.

## Parity forward

`parity_forward` uses `C - P = e^{-r tau}(F - K)` at the strike where the call and put mids are closest
(the VIX approach, §4.1), so `F = K + e^{r tau}(C - P)`. Only strikes with a two-sided call and a
two-sided put are candidates. The futures price enters put-call parity directly because the options are
European and written on the future.

The parity forward is used for pricing because the option and futures quotes are not perfectly
synchronous, and an error in `F` moves call and put implied vols in opposite directions. Checked on
July–September 2026 (64 days, 87,192 quotes, 452 chains): for options within 10% of the money and 14 to
60 days to expiry, the median absolute difference between call and put implied vol at the same strike is
0.08 vol points with the parity forward and 0.15 with the futures mid.

## Forward-gap diagnostic

The stage writes `reports/figures/forward_gap.png` (`forward_gap_figure`): the daily median of
`forward_gap` by maturity bucket and its distribution over all chains. `forward_gap_table` gives the
same data, one row per chain.

Findings on July–September 2026:
- Median gap -0.20 points, median absolute gap 1.10 points (about 1.4 basis points of `F`).
- 14 to 60 days to expiry: median absolute gap 0.75 points, 95th percentile 2.8 points.
- Over 120 days: median absolute gap 1.7 points, with gaps up to 15 points. These are options on
  back-month futures (`ESM7`, `ESU7`, `ESZ7`), whose futures quotes are wide.
- 28 of 452 chains had no parity forward and fall back to the futures mid. All but two are beyond
  60 days to expiry.

## Quotes without an implied volatility

- One-sided quotes have no `mid`, so no `iv` or Greeks (412 quotes).
- 296 two-sided quotes had a mid at or below discounted intrinsic value at the parity forward. All are
  in the money and most are within two weeks of expiry. Stage 2 tests intrinsic value against the
  futures mid, so a quote can pass there and fail here.

About 0.5% of two-sided quotes within 10% of the money have no implied vol.

## Known limitations

- The parity forward comes from a single strike. A bad quote at that strike moves the whole chain;
  the forward gap is the check. The chain-quality checks (roadmap 20, `docs/stages/02_snapshots.md`) run first
  in stage 4; they do not use the forward gap.
- `pricing.rate_source: fred` is not implemented; `risk_free_rate` raises for it.
- The findings above are from one quarter, not the full sample.
