"""Typed loader and validator for config/default.yaml.

Implements PLANNING.md §8 (Configuration).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing keys or has invalid values."""


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    product: str
    series: str
    start: date
    end: date
    snapshot_time_ct: str  # decided: 15:00 CT (§15.4)
    budget_usd: float


@dataclass(frozen=True)
class PricingConfig:
    rate_source: str
    rate_constant: float
    day_count: str


@dataclass(frozen=True)
class CleaningConfig:
    min_strikes: int
    min_range_pct: float
    drop_zero_bid: bool
    drop_crossed: bool


@dataclass(frozen=True)
class StripConfig:
    method: str
    price_col: str
    maturity_mode: str
    min_front_days: int


@dataclass(frozen=True)
class RealizedConfig:
    frequency: list[str]
    session: str  # decided: full_globex (§15.3)


@dataclass(frozen=True)
class HedgingConfig:
    hedge_delta: str
    rebalance: str
    moneyness_max: float
    maturity_buckets_days: list[int]


@dataclass(frozen=True)
class CostsConfig:
    option_halfspread: float | str
    hedge_slippage_ticks: float
    fee_per_contract_usd: float


@dataclass(frozen=True)
class InferenceConfig:
    hac_lags: int | str
    cluster_by: str


@dataclass(frozen=True)
class RegimesConfig:
    variable: str
    n_bins: int
    window: str


@dataclass(frozen=True)
class Config:
    data: DataConfig
    pricing: PricingConfig
    cleaning: CleaningConfig
    strip: StripConfig
    realized: RealizedConfig
    hedging: HedgingConfig
    costs: CostsConfig
    inference: InferenceConfig
    regimes: RegimesConfig


_CHOICES: dict[tuple[str, str], set[str]] = {
    ("data", "series"): {"weekly", "monthly_eom", "quarterly"},
    ("pricing", "rate_source"): {"constant", "fred"},
    ("pricing", "day_count"): {"ACT/365"},
    ("strip", "method"): {"vix_discrete", "iv_grid"},
    ("strip", "price_col"): {"mid", "bid", "ask"},
    ("strip", "maturity_mode"): {"matched", "constant_30d"},
    ("realized", "session"): {"full_globex", "rth"},
    ("hedging", "hedge_delta"): {"iv_daily", "iv_fixed", "rv"},
    ("hedging", "rebalance"): {"daily"},
    ("regimes", "variable"): {"sw_level", "trailing_rv_21d"},
    ("regimes", "window"): {"expanding"},
}


def _build(cls: type, section: str, raw: Any) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"section '{section}' must be a mapping")
    names = {f.name for f in fields(cls)}
    missing = names - raw.keys()
    extra = raw.keys() - names
    if missing:
        raise ConfigError(f"section '{section}' missing keys: {sorted(missing)}")
    if extra:
        raise ConfigError(f"section '{section}' has unknown keys: {sorted(extra)}")
    for (sec, key), allowed in _CHOICES.items():
        if sec == section and raw[key] not in allowed:
            raise ConfigError(f"{section}.{key}={raw[key]!r} not in {sorted(allowed)}")
    return cls(**raw)


def load_config(path: str | Path = DEFAULT_PATH) -> Config:
    """Load and validate a YAML config into typed dataclasses."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    sections = {f.name: f.type for f in fields(Config)}
    classes = {
        "data": DataConfig, "pricing": PricingConfig, "cleaning": CleaningConfig,
        "strip": StripConfig, "realized": RealizedConfig, "hedging": HedgingConfig,
        "costs": CostsConfig, "inference": InferenceConfig, "regimes": RegimesConfig,
    }
    missing = sections.keys() - raw.keys()
    if missing:
        raise ConfigError(f"missing sections: {sorted(missing)}")
    cfg = Config(**{name: _build(cls, name, raw[name]) for name, cls in classes.items()})
    _validate_values(cfg)
    return cfg


def _validate_values(cfg: Config) -> None:
    if cfg.data.start >= cfg.data.end:
        raise ConfigError("data.start must be before data.end")
    if cfg.data.budget_usd <= 0:
        raise ConfigError("data.budget_usd must be positive")
    if not 0 < cfg.hedging.moneyness_max < 1:
        raise ConfigError("hedging.moneyness_max must be in (0, 1)")
    if cfg.regimes.n_bins < 2:
        raise ConfigError("regimes.n_bins must be >= 2")
    if cfg.cleaning.min_strikes < 3:
        raise ConfigError("cleaning.min_strikes must be >= 3")
    if isinstance(cfg.inference.hac_lags, str) and cfg.inference.hac_lags != "auto":
        raise ConfigError("inference.hac_lags must be an int or 'auto'")
    if sorted(cfg.hedging.maturity_buckets_days) != cfg.hedging.maturity_buckets_days:
        raise ConfigError("hedging.maturity_buckets_days must be ascending")
