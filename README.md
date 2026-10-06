# es-vrp: The variance risk premium in E-mini S&P 500 futures options

**Question.** Do options on ES futures cost more than the volatility that follows, and can a delta-hedged
option seller collect that premium after costs?

We measure the variance risk premium (VRP) two ways and then test whether it is tradable:

1. **Carr and Wu (2009):** realized variance minus a model-free synthetic variance swap rate built from an
   out-of-the-money option strip.
2. **Bakshi and Kapadia (2003):** mean delta-hedged option gains, split by moneyness, maturity and volatility regime.
3. **Strategy:** a monthly short delta-hedged straddle with bid/ask, hedging slippage and fee costs, with tail risk reported.

Hypotheses, theory, data plan and the issue roadmap are in [docs/PLANNING.md](docs/PLANNING.md).
Every outcome is reportable: the premium exists and survives costs, exists but costs consume it, or the methods disagree.

> **Status:** scaffold. Results below will be filled in as issues in `docs/issues/` are completed.
> Only `pricing/black76.py` is implemented; other modules are stubs.

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

## Layout

See PLANNING.md §7. Code is in `src/esvrp/`, parameters in `config/default.yaml`, tests in `tests/`,
issue drafts in `docs/issues/`, open verification items in `docs/data.md`.

## Limitations

Short samples, exercise-style mismatch, data quality and costs; see PLANNING.md §4.8.
