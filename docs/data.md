# Data notes

Source: Databento `GLBX.MDP3`. See `docs/PLANNING.md` §5 for schemas, known issues and cost control.
Raw Databento data must never be committed (`data/` and `*.parquet` are gitignored); use
`src/esvrp/synthetic.py` for tests and demos.

## Open verification items

Items marked **[VERIFY]** or **[DECISION]** in `docs/PLANNING.md` are deliberately unresolved. Each has a
`TODO` comment at the point of use in the code.

| # | Type | Item | Plan § | Where in code |
|---|---|---|---|---|
| 1 | VERIFY | Exact identity and financing terms for hedged gain vs gamma-weighted `RV - SW`, before relying on it in the report | 4.5 | `hedging/simulator.py`, `vrp/strip.py` |
| 2 | VERIFY | Exercise style per ES option series (working assumption: weeklies European, quarterlies American) | 4.1 | `data/universe.py` |
| 3 | VERIFY | Authors/venue of "Exploring the Variance Risk Premium Across Assets"; confirm the S&P 500 result | 1, 3.3, 16 | docs only |
| 4 | VERIFY | Databento key can pull options; available history; cost for the date range | 5.1 | `data/client.py`, `config/default.yaml` |
| 5 | VERIFY | Schema names and availability for options (`definition`, `statistics`, `bbo-1m`, `ohlcv-1m`, `mbp-1`) | 5.2 | `data/client.py` |
| 6 | VERIFY | Settlement stat type exists for options in `statistics` | 5.2 | `data/underlying.py` |
| 7 | VERIFY | ES settlement time and snapshot cutoff (`snapshot_time_ct: 14:55`) | 5.2, 8 | `data/quotes.py`, `config.py` |
| 8 | VERIFY | `strike_price` display-factor issue does not affect the ES series; strikes bracket the futures price | 5.3 | `data/universe.py` |
| 9 | VERIFY | Data licensing terms for committing/sharing raw data | 5.3 | docs only |
| 10 | VERIFY | Options parent symbol family for ES (`data.product`) | 8 | `config.py`, `config/default.yaml` |
| 11 | DECISION | ES option series first: weekly, end-of-month or quarterly | 15.1 | `data/universe.py`, `config.py` |
| 12 | DECISION | Sample period; whether a longer daily-only sample exists | 15.2 | `config/default.yaml` |
| 13 | DECISION | Session for realized variance (full Globex vs RTH) and maintenance-gap treatment | 4.2, 15.3 | `data/underlying.py`, `vrp/realized.py`, `config.py` |
| 14 | DECISION | Snapshot time and settlement-time convention | 15.4 | `data/quotes.py` |
| 15 | DECISION | Strategy return denominator: notional, vega or margin | 4.7, 15.5 | `analysis/strategy.py` |
| 16 | DECISION | Primary VRP definition: difference, log ratio or swap return | 15.6 | `vrp/premium.py` |
| 17 | DECISION | Whether the CL/EIA extension is in scope | 15.7 | issue #20 |
