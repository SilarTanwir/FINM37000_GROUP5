"""Databento client wrapper with cost check and Parquet cache.

Implements PLANNING.md §5.4 (Cost and volume control). The only module allowed to read os.environ.

Every pull goes through `fetch`, which (1) returns the cached Parquet file if this exact request was
made before, otherwise (2) asks Databento for a cost estimate and logs it, (3) aborts if the estimate
exceeds the budget, and (4) pulls, caches and returns the data. Tests inject a fake client through the
`client` argument, so no test needs a key or the network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
ENV_PATH = REPO_ROOT / ".env"
API_KEY_VAR = "DATABENTO_API_KEY"


class BudgetExceededError(RuntimeError):
    """Raised when an estimated pull cost exceeds data.budget_usd."""


class MissingApiKeyError(RuntimeError):
    """Raised when DATABENTO_API_KEY is not set in the environment or .env."""


def get_api_key(env_path: Path | None = ENV_PATH) -> str:
    """Read the API key from the environment, loading `.env` first if it exists. Never log the key."""
    if env_path is not None and env_path.exists():
        load_dotenv(env_path, override=False)
    key = os.environ.get(API_KEY_VAR, "").strip()
    if not key:
        raise MissingApiKeyError(f"{API_KEY_VAR} is not set; add it to .env (see .env.example)")
    return key


def make_client(key: str | None = None) -> Any:
    """Build a `databento.Historical` client. Imported lazily so tests never touch the SDK."""
    import databento as db

    return db.Historical(key or get_api_key())


def _is_transient(err: Exception) -> bool:
    """Network timeouts, dropped connections, and Databento 5xx server errors (e.g. 504 gateway timeout)."""
    import requests

    if isinstance(err, (requests.Timeout, requests.ConnectionError)):
        return True
    status = getattr(err, "http_status", None)
    return isinstance(status, int) and status >= 500


def _with_retries(call: Any, *args: Any, attempts: int = 5, wait_s: float = 10.0, **kwargs: Any) -> Any:
    """Retry transient failures with a growing wait; anything else (e.g. a bad request) raises at once."""
    for attempt in range(1, attempts + 1):
        try:
            return call(*args, **kwargs)
        except Exception as err:
            if not _is_transient(err) or attempt == attempts:
                raise
            log.warning("Databento request failed (%s); retry %d/%d",
                        type(err).__name__, attempt, attempts - 1)
            time.sleep(wait_s * attempt)
    raise AssertionError("unreachable")


def _normalize_symbols(symbols: str | list[str]) -> list[str]:
    return sorted([symbols] if isinstance(symbols, str) else list(symbols))


def _as_str(d: str | date | datetime) -> str:
    return d.isoformat() if isinstance(d, (date, datetime)) else str(d)


def request_params(
    dataset: str, schema: str, symbols: str | list[str],
    start: str | date | datetime, end: str | date | datetime, stype_in: str = "raw_symbol",
) -> dict[str, Any]:
    """Canonical request description: symbol order and date types do not change the cache key."""
    return {
        "dataset": dataset,
        "schema": schema,
        "symbols": _normalize_symbols(symbols),
        "start": _as_str(start),
        "end": _as_str(end),
        "stype_in": stype_in,
    }


def cache_key(params: dict[str, Any]) -> str:
    """Short, stable hash of the request parameters (§5.4.3)."""
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def estimate_cost(
    dataset: str, schema: str, symbols: str | list[str],
    start: str | date | datetime, end: str | date | datetime,
    stype_in: str = "raw_symbol", client: Any = None,
) -> float:
    """Return Databento's estimated cost in USD; log it before every pull (§5.4.1)."""
    params = request_params(dataset, schema, symbols, start, end, stype_in)
    client = client or make_client()
    cost = float(_with_retries(client.metadata.get_cost, **params))
    log.info("Databento cost estimate: $%.2f for %s", cost, params)
    return cost


def fetch(
    dataset: str, schema: str, symbols: str | list[str],
    start: str | date | datetime, end: str | date | datetime,
    budget_usd: float, stype_in: str = "raw_symbol",
    cache_dir: Path = DATA_DIR, client: Any = None,
) -> pd.DataFrame:
    """Cost-check, then pull (or load from the Parquet cache keyed by request parameters).

    The budget is a per-request hard stop (§5.4.4). What the budget should mean on the
    program's Plus plan is an open item in docs/data.md.
    """
    params = request_params(dataset, schema, symbols, start, end, stype_in)
    key = cache_key(params)
    data_path = Path(cache_dir) / f"{key}.parquet"
    meta_path = Path(cache_dir) / f"{key}.json"

    if data_path.exists():
        log.info("Cache hit %s for %s", data_path.name, params)
        return pd.read_parquet(data_path)

    client = client or make_client()
    cost = estimate_cost(dataset, schema, symbols, start, end, stype_in, client=client)
    if cost > budget_usd:
        raise BudgetExceededError(
            f"Estimated cost ${cost:.2f} exceeds budget ${budget_usd:.2f} for {params}"
        )

    df = _with_retries(client.timeseries.get_range, **params).to_df()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_path)
    meta_path.write_text(json.dumps({**params, "estimated_cost_usd": cost, "rows": len(df)}, indent=2))
    log.info("Pulled %d rows into %s", len(df), data_path.name)
    return df
