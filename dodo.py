"""doit task graph mirroring pipeline stages 1-8 (PLANNING.md §6).

Run `doit list` to see tasks, `doit test` for the offline test suite. Stage actions call
`python -m esvrp.stages <n>` which is not yet implemented (stubs raise NotImplementedError).
"""

DOIT_CONFIG = {"default_tasks": ["test"], "verbosity": 2}

CFG = "config/default.yaml"
D = "data"
R = "reports"


def _stage(n, name, file_dep, targets, doc):
    return {
        "name": name,
        "actions": [f"python -m esvrp.stages {n}"],
        "file_dep": [CFG, *file_dep],
        "targets": targets,
        "doc": doc,
    }


def task_universe():
    """Stage 1: option universe from definitions."""
    return _stage(1, "universe", ["src/esvrp/data/universe.py"], [f"{D}/universe.parquet"], "Stage 1")


def task_quotes():
    """Stage 2: cleaned daily quote snapshots."""
    return _stage(2, "quotes", [f"{D}/universe.parquet", "src/esvrp/data/quotes.py"],
                  [f"{D}/quotes.parquet"], "Stage 2")


def task_underlying():
    """Stage 3: 5-minute and daily underlying futures series."""
    return _stage(3, "underlying", [f"{D}/universe.parquet", "src/esvrp/data/underlying.py"],
                  [f"{D}/underlying_5m.parquet", f"{D}/underlying_daily.parquet"], "Stage 3")


def task_chain():
    """Stage 4: chain with forward, tau, mid, IV, delta, moneyness."""
    return _stage(4, "chain",
                  [f"{D}/quotes.parquet", f"{D}/underlying_daily.parquet",
                   "src/esvrp/pricing/black76.py", "src/esvrp/pricing/forward.py"],
                  [f"{D}/chain.parquet"], "Stage 4")


def task_vrp():
    """Stage 5: variance swap strip, realized variance, VRP."""
    return _stage(5, "vrp",
                  [f"{D}/chain.parquet", f"{D}/underlying_5m.parquet", f"{D}/underlying_daily.parquet",
                   "src/esvrp/vrp/strip.py", "src/esvrp/vrp/realized.py", "src/esvrp/vrp/premium.py"],
                  [f"{D}/strip.parquet", f"{D}/rv.parquet", f"{D}/vrp.parquet"], "Stage 5")


def task_hedging():
    """Stage 6: delta-hedged gains per option."""
    return _stage(6, "hedging",
                  [f"{D}/chain.parquet", f"{D}/underlying_daily.parquet",
                   "src/esvrp/hedging/simulator.py"],
                  [f"{D}/hedged_gains.parquet"], "Stage 6")


def task_analysis():
    """Stage 7: inference tables and strategy backtest."""
    return _stage(7, "analysis",
                  [f"{D}/vrp.parquet", f"{D}/hedged_gains.parquet", "src/esvrp/hedging/costs.py",
                   "src/esvrp/analysis/inference.py", "src/esvrp/analysis/strategy.py",
                   "src/esvrp/analysis/buckets.py"],
                  [f"{R}/tables/summary.csv", f"{R}/tables/strategy.csv"], "Stage 7")


def task_report():
    """Stage 8: figures and tables."""
    return _stage(8, "report",
                  [f"{R}/tables/summary.csv", f"{R}/tables/strategy.csv",
                   "src/esvrp/reporting/figures.py", "src/esvrp/reporting/tables.py"],
                  [f"{R}/figures/.done"], "Stage 8")


def task_test():
    """Run the offline test suite and ruff."""
    return {"actions": ["ruff check .", "pytest -q"], "verbosity": 2}
