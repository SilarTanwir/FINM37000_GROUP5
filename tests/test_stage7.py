"""Stage 7 end to end on simulated chains: H3 and the strategy (roadmaps 15 and 16)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from test_hedged_gains import make_chain, make_daily

from esvrp import config
from esvrp.analysis import pipeline
from esvrp.hedging.simulator import build_hedged_gains


def test_stage7_writes_tables_and_figures(tmp_path):
    chain = make_chain()
    gains, _ = build_hedged_gains(chain, make_daily())
    vrp = chain.groupby(["date", "expiry"]).size().rename("n").reset_index()
    vrp["vrp_diff"] = np.linspace(-0.02, 0.01, len(vrp))
    vrp["sw"], vrp["rv"] = 0.04, 0.04 + vrp["vrp_diff"]
    vrp["tau"] = 30 / 365  # stage 5 carries each chain's tau; the H2 regimes pick the chain nearest 30 days
    gains.to_parquet(tmp_path / "hedged_gains.parquet")
    vrp.to_parquet(tmp_path / "vrp.parquet")

    out = pipeline.run(config.load_config(), data_dir=tmp_path, report_dir=tmp_path / "reports")

    for f in ("tables/h3.csv", "tables/h3_sensitivity.csv", "tables/h3_disagreements.csv",
              "tables/strategy.csv", "tables/strategy_trades.csv", "tables/strategy_structures.csv",
              "tables/strategy_worst_months.csv", "tables/strategy_margin.csv", "tables/summary.csv",
              "figures/h3.png",
              "figures/strategy.png"):
        assert (tmp_path / "reports" / f).stat().st_size > 0, f
    assert (tmp_path / "h3.parquet").exists()
    assert len(out["strategy_trades"]) == 1  # September is the only expiry inside the sample
    assert list(out["strategy_margin"].index) == ["6%", "8%", "10%"]
    structures = out["strategy_structures"]
    assert structures.index[0] == "straddle (ATM)" and len(structures) == 4
    assert len(out["strategy_worst_months"]) == 1 and out["strategy_worst_months"].iloc[0]["rank"] == 1
    assert out["h3"].loc["all chains (overlapping)", "n"] == len(out["h3_chains"]) > 0
    assert pd.read_csv(tmp_path / "reports/tables/strategy.csv", index_col=0).loc["net", "n"] == 1


def test_stage7_figures_handle_no_data(tmp_path):
    from esvrp.analysis.strategy import run_strategy
    from esvrp.reporting.figures import h3_figure, strategy_figure

    res = run_strategy(pd.DataFrame(columns=["hedge_delta"]))
    assert strategy_figure(res.trades, res.metrics, tmp_path / "s.png").exists()
    assert h3_figure(pd.DataFrame(columns=["date", "expiry", "vrp", "hg", "tau"]), pd.DataFrame(),
                     tmp_path / "h.png").exists()


def test_stage_runner_and_task_graph_include_stage7():
    import esvrp.stages as stages

    assert "esvrp.analysis.pipeline" in Path(stages.__file__).read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("dodo", Path(__file__).parents[1] / "dodo.py")
    dodo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dodo)
    task = dodo.task_analysis()
    assert task["actions"] == ["python -m esvrp.stages 7"]
    assert "reports/tables/h3.csv" in task["targets"]
    tasks = [getattr(dodo, name)() for name in dir(dodo) if name.startswith("task_")]
    targets = {t for task in tasks for t in task.get("targets", [])}
    deps = {d for task in tasks for d in task.get("file_dep", [])}
    produced_or_source = targets | {d for d in deps if not d.startswith(("data/", "reports/"))}
    assert deps <= produced_or_source  # every data file a task needs is some task's target


def test_stage_runner_and_task_graph_include_stage8():
    import esvrp.stages as stages

    assert "esvrp.reporting.tables" in Path(stages.__file__).read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("dodo", Path(__file__).parents[1] / "dodo.py")
    dodo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dodo)
    task = dodo.task_report()
    assert task["actions"] == ["python -m esvrp.stages 8"]
    assert "reports/tables/headline.csv" in task["targets"]
    assert {"reports/tables/h1.csv", "reports/tables/summary.csv"} <= set(task["file_dep"])
