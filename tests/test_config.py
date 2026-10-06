"""Config loader validates config/default.yaml (PLANNING.md §8)."""

import pytest

from esvrp.config import ConfigError, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.data.dataset == "GLBX.MDP3"
    assert cfg.hedging.maturity_buckets_days == [14, 30, 60]


def test_bad_choice_rejected(tmp_path):
    text = open("config/default.yaml").read().replace("price_col: mid", "price_col: wat")
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    with pytest.raises(ConfigError):
        load_config(p)
