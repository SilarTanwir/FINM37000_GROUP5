# Stage 7: analysis

See `docs/PLANNING.md` §6 for inputs, outputs and the theory sections (§4.5–4.7). Code:
`src/esvrp/analysis/reconcile.py` (H3, roadmap 15), `src/esvrp/analysis/strategy.py` (backtest, roadmap
16), `src/esvrp/analysis/inference.py` (standard errors), `src/esvrp/analysis/pipeline.py` (stage entry
point). Run with `doit analysis` or `python -m esvrp.stages 7` (needs `data/vrp.parquet` from stage 5 and
`data/hedged_gains.parquet` from stage 6).

Not yet run on real data: stage 5 (PR #36) is not merged, and no chain has been built in this checkout.
The moneyness × maturity × regime table (roadmap 14, `reports/tables/summary.csv`) is not written yet.

## Outputs

| File | Content |
|---|---|
| `data/h3.parquet` | One row per chain with both a premium and a hedged gain: `vrp`, `sw`, `rv`, `hg`, `iv2_gw`, `rv_gw`, `n_options`, `tau` |
| `reports/tables/h3.csv` | H3 statistics for every chain and for the non-overlapping monthly subsample |
| `reports/tables/strategy.csv` | Risk metrics of the monthly short straddle: per unit vega, dollars and return on margin, each gross and net |
| `reports/tables/strategy_trades.csv` | One row per monthly trade |
| `reports/figures/h3.png` | VRP against the hedged gain (planned output 5) |
| `reports/figures/strategy.png` | Cumulative gross and net P&L, and each month's net P&L (planned output 6) |

## H3: reconciling the VRP with hedged gains (roadmap 15)

**The two measures, in the same units.** The VRP is `RV - SW` per chain (stage 5, each expiry at its own
maturity), in annualized variance. For the hedged gains, hedging at a fixed implied vol `sigma_i`,

`Pi ~ 1/2 sum_n Gamma_n F_n^2 (r_{n+1}^2 - sigma_i^2 dt_n)`   (§4.5)

so `HG = sum Pi / sum (1/2 sum_n Gamma_n F_n^2 dt_n)` over a chain's options is a gamma-weighted average of
realized minus implied variance, also annualized. Stage 6 stores the weight as `gamma_dt` for every option
and hedge delta; `chain_hedged_variance` sums gains and weights per chain and divides. H3 uses the
`iv_fixed` hedge, the case the identity describes.

Sum, then divide: averaging each option's own ratio (`var_gap` in stage 6) is biased upward. On simulated
paths with realized vol equal to implied vol (18%), the mean of the per-option ratios is +0.0014 at daily
hedging (falling to +0.0002 at 16 hedges a day), while the ratio of the sums is −0.0003 ± 0.0002.

**Statistics** (`reconcile`), for every chain and for one chain per expiry (the first date in the
expiry's month, the strategy's entry rule, so windows do not overlap):
- `mean_vrp`, `mean_hg` with naive and HAC t-statistics (lag 30 on the daily series, 6 on the monthly).
- `sign_agree`: share of chains where VRP and HG have the same sign.
- `pearson` with a 95% interval, and `spearman`. On every chain the interval is a block bootstrap over
  dates (all chains of a date together, runs of 21 trading days), because overlapping windows share most
  of their returns and an interval that treats chains as independent is too narrow (checked in
  `test_reconcile`). On the monthly series it is Fisher's z.

### How to read the result

H3 holds if the two series **agree in sign on average and co-move**: means of the same sign, and a
correlation whose interval excludes zero, on the monthly series as well as on every chain. It is not a
test of equality, and the two should not be expected to match:

- `HG - VRP = (RV_gw - RV) - (IV2_gw - SW)`. `IV2_gw` is the gamma-weighted implied variance of the
  chain's options, which sits near at-the-money implied vol. `SW` weights every strike by `1/K^2` and
  picks up the put skew, so `SW > IV2_gw` and **HG should be less negative than VRP**. The gap
  `SW - IV2_gw` is part of the result: it is the share of the premium paid for the skew, which a
  near-the-money hedged position does not earn.
- `RV_gw` weights each day's squared return by the option's gamma on that day, which rises as the
  futures price nears the strike and expiry nears. `RV` weights every day equally. They differ most
  when the futures price moves far from the money.
- Discrete daily hedging adds noise to `HG` path by path but no bias on average (`test_hedge_identity`,
  `test_gamma_weighted_gain_recovers_the_variance_gap`).
- Bakshi and Kapadia's own link is vega-weighted, not gamma-weighted (§4.5). Agreement here supports the
  two papers measuring the same premium; disagreement is reportable (§1).

Possible outcomes and what each means:
- Same sign, correlated: the two methods measure one premium; the size gap is the skew share.
- Same sign, weak correlation: both find a premium on average, but month-to-month noise (discrete
  hedging, the weighting differences above) dominates. Compare the monthly and daily correlations.
- Different signs: the premium is concentrated in the strikes the strip sees and the hedged options do
  not (deep out-of-the-money puts), or one measure has a data problem. Check `sw` truncation flags and
  the forward gap first.

Results on the real sample: to be written once stages 5–7 have run.

## Strategy backtest (roadmap 16)

**Rule** (§6 stage 7). For each end-of-month expiry, on the first snapshot of that month, sell one
delta-hedged straddle at the strike closest to the forward that has both a call and a put; hold to expiry;
hedge daily with the option's own futures contract. Trades never overlap: each expires at 15:00 CT on the
last business day of its month and the next starts on the next month's first snapshot.

**P&L.** The short earns minus each leg's long gain from stage 6 (`hedging.hedge_delta`, baseline
`iv_daily`). Selling at the bid is the mid gain less the half-spread, which is in each leg's `cost` along
with 0.5 tick of slippage per rebalance and $1.50 per contract on every trade. `gross` is at mid, `net` after
costs; `cost_multiplier` scales costs for the robustness suite (roadmap 17).

**Returns** (decided, §4.7 and §15.5):
- Per unit of entry vega, primary: P&L divided by the straddle's dollar vega per vol point
  (`vega × $50 × 0.01`). For a seller this is about implied minus realized vol in vol points.
- On an assumed margin, secondary: P&L over 10% of the two legs' futures notional (`MARGIN_FRACTION` in
  `pipeline.py`). This is an assumption, as §4.7 says; read drawdowns on it with that in mind.
- Dollars for one straddle, for scale.

**Metrics** (`risk_metrics`, one value per monthly trade): mean with naive and HAC (lag 6)
t-statistics, annualized Sharpe (`× sqrt(12)`), skew, share of winning months, the worst two and the best
month, maximum drawdown of the running sum (from a starting peak of zero) and CVaR, the mean of the worst
5% of months (at least one month, so with 36 trades it is the worst month).

Not covered: worst day (P&L is only measured at expiry, not marked daily), the strangle variant (roadmap
17), and option spreads by moneyness bucket from the `mbp-1` sample (roadmap 13); the entry half-spread is
taken from each quote's own bid and ask.

## Checks

- `test_reconcile`: chain aggregation sums before dividing; the monthly subsample takes the first date
  in each expiry's month; linked series give a high correlation with an interval above zero, unrelated
  series an interval covering zero; the date-block interval is wider than an iid one on overlapping data;
  unmatched and missing chains are dropped.
- `test_strategy`: one trade a month on the first snapshot, at the money, with both legs; short P&L, costs
  and every normalization by hand; cost multiplier; risk metrics by hand, including drawdown from the
  start.
- `test_stage7`: the stage on simulated chains writes every table and figure; figures with no data; the
  stage runner and the `doit` graph.
