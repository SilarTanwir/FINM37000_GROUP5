"""Stage 5 end to end (roadmap 9-11 wiring): synthetic chains and bars in, tables and figures out."""

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from esvrp.config import load_config
from esvrp.data.underlying import build_bars
from esvrp.vrp.pipeline import run

TZ = "America/Chicago"
SNAPS = ["2026-10-12", "2026-10-13"]  # two snapshot days, 15:00 CT
EXPIRY_DAYS = [1, 3]  # two expiries per day, in days after the snapshot


@pytest.fixture
def stage_inputs(tmp_path, raw_week, flat_vol_chain):
    raw = raw_week()
    bars_5m = build_bars(raw, "5m", "full_globex")
    daily = build_bars(raw, "daily", "full_globex").assign(settle=lambda d: d["close"])
    frames = []
    for day in SNAPS:
        snap = pd.Timestamp(f"{day} 15:00", tz=TZ).tz_convert("UTC")
        for days in EXPIRY_DAYS:
            tau = days / 365
            chain, F, r, _ = flat_vol_chain(sigma=0.2, tau=tau)
            frames.append(chain.assign(
                date=pd.Timestamp(day).date(), expiry=snap + pd.Timedelta(days=days), underlying="ESZ6",
                snapshot_ts=snap, F=F, r=r, tau=tau,
            ))
    data = tmp_path / "data"
    data.mkdir()
    pd.concat(frames, ignore_index=True).to_parquet(data / "chain.parquet")
    bars_5m.to_parquet(data / "underlying_5m.parquet")
    daily.to_parquet(data / "underlying_daily.parquet")
    return data, tmp_path / "reports"


def _cfg(min_front_days=1):
    cfg = load_config()
    return dataclasses.replace(cfg, strip=dataclasses.replace(cfg.strip, min_front_days=min_front_days))


def test_stage5_writes_every_table_and_figure(stage_inputs):
    data, reports = stage_inputs
    tables = run(_cfg(), data, reports)
    for name in ["strip", "rv", "vrp", "strip_cm", "rv_cm", "vrp_cm"]:
        assert (data / f"{name}.parquet").exists()
        assert len(pd.read_parquet(data / f"{name}.parquet")) == len(tables[name])
    assert (reports / "tables" / "vrp_descriptives.csv").exists()
    assert (reports / "tables" / "vrp_cm_descriptives.csv").exists()
    for fig in ["sw_vs_rv", "vrp_distribution", "rv_comparison", "cm_sw_vs_rv", "cm_vrp_distribution"]:
        assert (reports / "figures" / f"{fig}.png").stat().st_size > 5000


def test_stage5_tables_have_one_row_per_chain_and_flat_vol_swap_rate(stage_inputs):
    data, reports = stage_inputs
    t = run(_cfg(), data, reports)
    assert len(t["strip"]) == len(t["rv"]) == len(t["vrp"]) == len(SNAPS) * len(EXPIRY_DAYS)
    assert t["strip"]["sw_mid"].to_numpy() == pytest.approx(0.2**2, rel=0.01)
    v = t["vrp"]
    assert (v["vrp_diff"] == v["rv"] - v["sw"]).all()
    assert v["rv"].notna().all() and (v["end_gap_s"] == 0).all()  # the windows end on bars


def test_stage5_constant_maturity_flags_extrapolation_and_missing_window(stage_inputs):
    data, reports = stage_inputs
    t = run(_cfg(), data, reports)
    cm = t["strip_cm"]
    assert len(cm) == len(SNAPS) and cm["extrapolated"].all()  # both expiries are before 30 days
    assert cm["sw_mid"].to_numpy() == pytest.approx(0.2**2, rel=0.01)  # a flat swap rate stays flat
    # The simulated bars end days before snapshot + 30 days: the gap column says so.
    assert (t["rv_cm"]["end_gap_s"] > 20 * 86400).all()


def test_stage5_without_two_valid_expiries_has_no_constant_maturity(stage_inputs):
    data, reports = stage_inputs
    t = run(_cfg(min_front_days=8), data, reports)  # every expiry is under 8 days away
    assert t["strip_cm"].empty and t["vrp_cm"].empty
    assert len(t["vrp"]) == len(SNAPS) * len(EXPIRY_DAYS)
    assert not (reports / "figures" / "cm_sw_vs_rv.png").exists()


def test_stage5_rejects_unimplemented_method_and_empty_chain(stage_inputs):
    data, reports = stage_inputs
    cfg = load_config()
    with pytest.raises(NotImplementedError):
        run(dataclasses.replace(cfg, strip=dataclasses.replace(cfg.strip, method="iv_grid")), data, reports)
    pd.read_parquet(data / "chain.parquet").iloc[:0].to_parquet(data / "chain.parquet")
    with pytest.raises(ValueError):
        run(cfg, data, reports)


def test_stage5_is_wired_into_the_stage_runner(stage_inputs, monkeypatch):
    import esvrp.stages as stages
    import esvrp.vrp.pipeline as pipeline

    called = []
    monkeypatch.setattr(pipeline, "run", lambda cfg: called.append(cfg))
    stages.run_stage(5)
    assert len(called) == 1


def test_dodo_stage5_task_lists_the_outputs():
    # Load dodo.py by path: the repo root is not on sys.path when CI runs plain `pytest`.
    spec = importlib.util.spec_from_file_location("dodo", Path(__file__).resolve().parents[1] / "dodo.py")
    dodo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dodo)

    task = dodo.task_vrp()
    assert "python -m esvrp.stages 5" in task["actions"][0]
    assert {"data/strip.parquet", "data/rv.parquet", "data/vrp.parquet"} <= set(task["targets"])
    assert np.all([f.startswith(("data/", "reports/", "src/", "config/")) for f in task["file_dep"]])
