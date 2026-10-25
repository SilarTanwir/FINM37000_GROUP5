> Developer notes: the planning README as of 25 Oct 2026, kept for reference. The project report is [README.md](../README.md).

# es-vrp: The variance risk premium in E-mini S&P 500 futures options

**Question.** Do options on ES futures cost more than the volatility that follows, and can a delta-hedged
option seller collect that premium after costs?

We measure the variance risk premium (VRP) two ways and then test whether it is tradable:

1. **Carr and Wu (2009):** realized variance minus a model-free synthetic variance swap rate built from an
   out-of-the-money option strip.
2. **Bakshi and Kapadia (2003):** mean delta-hedged option gains, split by moneyness, maturity and volatility regime.
3. **Strategy:** a monthly short delta-hedged straddle with bid/ask, hedging slippage and fee costs, with tail risk reported.

Hypotheses, data plan and the issue roadmap are in [docs/PLANNING.md](PLANNING.md); the math is
derived in [docs/theory.md](theory.md); data assumptions are in [docs/data.md](data.md).
Every outcome is reportable: the premium exists and survives costs, exists but costs consume it, or the methods disagree.

> **Status (20 Oct 2026):** stages 1–5 are implemented and merged on `main`; stages 6–7 (hedged gains,
> H3 reconciliation, strategy backtest) are in PR #43. Everything is tested on simulated data. The first
> full real-data run is in progress (issue #35); results will be added here after it.

| # | Stage | Command | Main output | Code | Status |
|---|---|---|---|---|---|
| 1 | Universe | `doit universe` | `data/universe.parquet` | `data/universe.py` | merged |
| 2 | Snapshots | `doit quotes` | `data/quotes.parquet` | `data/quotes.py` | merged |
| 3 | Underlying | `doit underlying` | `data/underlying_5m.parquet`, `underlying_daily.parquet` | `data/underlying.py` | merged |
| 4 | Chain | `doit chain` | `data/chain.parquet` | `pricing/chain.py` | merged |
| 5 | VRP | `doit vrp` | `data/strip.parquet`, `rv.parquet`, `vrp.parquet` | `vrp/` | merged |
| 6 | Hedging | `doit hedging` | `data/hedged_gains.parquet` | `hedging/simulator.py` | PR #43 |
| 7 | Analysis | `doit analysis` | `reports/tables/h3.csv`, `strategy.csv` | `analysis/` | PR #43 |
| 8 | Report | `doit report` | `reports/` | `reporting/` | not started |

Each stage is documented in `docs/stages/` (inputs, outputs, rules, checks).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add DATABENTO_API_KEY only if you will pull real data
```

## Run

No Databento key is needed for tests or synthetic-data runs.

```bash
ruff check .
pytest -q
doit list        # pipeline stages 1-8
```

Real-data runs need a key and a budget (`data.budget_usd` in `config/default.yaml`); every pull logs its
estimated cost first and aborts if over budget. Raw Databento data is never committed.

### Real-data run

```bash
cp .env.example .env     # set DATABENTO_API_KEY
doit universe            # stage 1: EW option definitions (weekly snapshots)
doit quotes              # stage 2: 15:00 CT quotes, one business day at a time
doit underlying          # stage 3: 1-minute bars and settlements per futures contract
doit chain               # stage 4: no data pulled
doit vrp                 # stage 5: no data pulled
```

- **Cost:** every pull for the Oct 2023–Sep 2026 sample (`definition`, `bbo-1m`, `ohlcv-1m`, `statistics`
  on `GLBX.MDP3`) is estimated at $0 on the program's Plus plan.
- **Runtime:** stage 3 took about 6 minutes on a test universe (19 contracts); stage 2 makes two requests
  per business day. Pulls are cached in `data/`, so a rerun only pulls what is missing.
- **What to check:** each stage logs its row counts; stage 2 writes `data/quotes_cleaning_log.csv`;
  stage 4 writes `reports/figures/forward_gap.png`; stage 5 writes the SW vs RV figures. `doit` reruns a
  stage only when its inputs or the config change (`doit forget <task>` forces a rerun).

## Layout

See PLANNING.md §7. Code is in `src/esvrp/`, parameters in `config/default.yaml`, tests in `tests/`,
task source text in `docs/issues/` (tracked as GitHub Issues), data assumptions and open and resolved
decisions in `docs/data.md`, derivations in `docs/theory.md`, one document per stage in `docs/stages/`.

## Limitations

Short samples, exercise-style mismatch, data quality and costs; see PLANNING.md §4.8.

## Team

FINM 37000 Project Group 5. Roles for Project Setup (PLANNING.md §11):

| Member | GitHub | Role |
|---|---|---|
| Tanwir Silar | [@SilarTanwir](https://github.com/SilarTanwir) | Tech leader (owns the main repository) |
| Alex Nikolaev | [@alexnikolaevh](https://github.com/alexnikolaevh) | Communication leader |
| Aryaa Gunavante | [@agunavante](https://github.com/agunavante) | Design leader |
| Andrew Yang | [@andrewyang27](https://github.com/andrewyang27) | Design leader |
