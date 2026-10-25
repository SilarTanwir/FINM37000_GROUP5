"""Stage 8 (part of #21): headline table and figure check."""

import logging

import pandas as pd
import pytest

from esvrp.reporting.figures import STAGE_FIGURES, make_figures
from esvrp.reporting.tables import HEADLINE_COLUMNS, headline_table, run


def write_inputs(tables):
    tables.mkdir(parents=True)
    pd.DataFrame({"sample": ["matched, monthly (non-overlapping)"] * 2, "premium": ["vrp_diff", "vrp_log"],
                  "n": [30, 30], "mean": [-0.007, -0.5], "t_naive": [-2.0, -7.0], "t_hac": [-1.6, -6.4],
                  "t_hac_30": [None, None], "sharpe_short": [None, 1.2]},
                 ).to_csv(tables / "h1.csv", index=False)
    rows = []
    for m in ("otm_put", "near_atm", "otm_call"):
        for reg in ("all", "low", "mid", "high"):
            for meas in ("per_vega", "gain_over_F"):
                rows.append({"moneyness": m, "maturity": "14-30d", "regime": reg, "measure": meas, "n": 10,
                             "n_dates": 5, "mean": -1.0 if m == "otm_put" else 0.0, "t_naive": -3.0,
                             "t_hac": -2.0, "t_cluster": -2.5})
    pd.DataFrame(rows).to_csv(tables / "summary.csv", index=False)
    pd.DataFrame({"n": [100, 30], "pearson": [0.5, 0.9], "sign_agree": [0.8, 0.9], "ci_method": ["a", "b"]},
                 index=["all chains (overlapping)", "monthly (non-overlapping)"]).to_csv(tables / "h3.csv")
    pd.DataFrame({"n": [32, 32, 32], "mean": [-0.16, -1.01, -500.0], "sharpe": [-0.1, -0.78, -0.7],
                  "units": ["vol points", "vol points", "dollars"]},
                 index=["gross_per_vega", "net_per_vega", "gross"]).to_csv(tables / "strategy.csv")


def test_headline_collects_each_hypothesis(tmp_path):
    write_inputs(tmp_path / "reports" / "tables")
    out = run(None, report_dir=tmp_path / "reports")
    h = pd.read_csv(tmp_path / "reports" / "tables" / "headline.csv")
    assert list(h.columns) == HEADLINE_COLUMNS
    assert set(h["hypothesis"]) == {"H1", "H2", "H3", "H4"}
    assert len(out["headline"]) == len(h)

    def value(hyp, row, stat):
        sel = h[(h["hypothesis"] == hyp) & (h["row"] == row) & (h["statistic"] == stat)]
        assert len(sel) == 1
        return sel["value"].iloc[0]

    assert value("H1", "matched, monthly (non-overlapping) | vrp_log", "t_hac") == -6.4
    # H2: pooled regime, per vega only.
    assert value("H2", "otm_put | 14-30d", "mean") == -1.0
    assert len(h[h["hypothesis"] == "H2"]) == 3 * 6
    assert value("H3", "monthly (non-overlapping)", "pearson") == 0.9
    assert value("H4", "net_per_vega", "sharpe") == -0.78
    # Only the per-vega and net rows of the strategy; text columns never appear.
    assert set(h.loc[h["hypothesis"] == "H4", "row"]) == {"gross_per_vega", "net_per_vega"}
    assert not h["statistic"].isin(["units", "ci_method"]).any()


def test_missing_inputs_are_skipped_with_a_warning(tmp_path, caplog):
    tables = tmp_path / "reports" / "tables"
    write_inputs(tables)
    (tables / "summary.csv").unlink()
    with caplog.at_level(logging.WARNING):
        run(None, report_dir=tmp_path / "reports")
    h = pd.read_csv(tables / "headline.csv")
    assert "H2" not in set(h["hypothesis"])
    assert "summary.csv not found" in caplog.text
    assert "robustness.csv not found" in caplog.text


def test_no_inputs_writes_an_empty_table(tmp_path):
    run(None, report_dir=tmp_path / "reports")
    h = pd.read_csv(tmp_path / "reports" / "tables" / "headline.csv")
    assert h.empty and list(h.columns) == HEADLINE_COLUMNS
    assert (tmp_path / "reports" / "figures" / ".done").exists()


def test_robustness_rows_are_labelled_by_their_text_columns():
    rb = pd.DataFrame({"variant": ["cost x2", "cost x0.5"], "hypothesis": ["H4", "H4"], "mean": [-2.0, -0.5]})
    h = headline_table({"robustness": rb})
    assert h["row"].tolist() == ["cost x2 | H4", "cost x0.5 | H4"]
    assert h["value"].tolist() == [-2.0, -0.5]


def test_make_figures_lists_found_figures_and_warns_about_missing(tmp_path, caplog):
    (tmp_path / "h3.png").write_bytes(b"png")
    with caplog.at_level(logging.WARNING):
        found = make_figures({}, tmp_path)
    assert found == [tmp_path / "h3.png"]
    assert (tmp_path / ".done").read_text(encoding="utf-8") == "h3.png\n"
    assert "strategy.png" in caplog.text
    assert len(STAGE_FIGURES) == 7


@pytest.mark.parametrize("name", STAGE_FIGURES)
def test_every_stage_figure_is_drawn_by_some_stage(name):
    """Guards the list against renames: each file name appears in the stage code that writes it."""
    import pathlib

    src = pathlib.Path(__file__).parents[1] / "src" / "esvrp"
    text = "".join(p.read_text(encoding="utf-8") for p in src.rglob("*.py") if p.name != "figures.py")
    assert name in text
