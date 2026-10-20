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
    backfill_minutes: int  # look-back for strikes with no quote at the snapshot minute (§6 stage 2)


@dataclass(frozen=True)
class QualityConfig:
    action: str  # flag | reject: what to do with a chain that fails a threshold (§6 stage 4, roadmap 20)
    min_calls: int  # two-sided calls per chain
    min_puts: int  # two-sided puts per chain
    min_strikes: int
    min_range_pct: float  # strikes must reach F(1 - x) and F(1 + x)
    max_quote_age_s: float  # a quote older than this at the snapshot is stale
    max_stale_share: float  # flag the chain if more than this share of its quotes is stale
    max_futures_age_s: float  # futures observation older than this at the snapshot
    max_sync_gap_s: float  # median |option quote time - futures observation time|


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
    quality: QualityConfig
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
    ("quality", "action"): {"flag", "reject"},
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
            "quality": QualityConfig,
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
    q = cfg.quality
    if min(q.min_calls, q.min_puts, q.min_strikes) < 1:
        raise ConfigError("quality.min_calls, min_puts and min_strikes must be >= 1")
    if not 0 < q.min_range_pct < 1:
        raise ConfigError("quality.min_range_pct must be in (0, 1)")
    if not 0 <= q.max_stale_share <= 1:
        raise ConfigError("quality.max_stale_share must be in [0, 1]")
    if min(q.max_quote_age_s, q.max_futures_age_s, q.max_sync_gap_s) < 0:
        raise ConfigError("quality age and sync thresholds must be >= 0")
    if cfg.regimes.n_bins < 2:
        raise ConfigError("regimes.n_bins must be >= 2")
    if cfg.cleaning.min_strikes < 3:
        raise ConfigError("cleaning.min_strikes must be >= 3")
    if isinstance(cfg.inference.hac_lags, str) and cfg.inference.hac_lags != "auto":
        raise ConfigError("inference.hac_lags must be an int or 'auto'")
    if sorted(cfg.hedging.maturity_buckets_days) != cfg.hedging.maturity_buckets_days:
        raise ConfigError("hedging.maturity_buckets_days must be ascending")
