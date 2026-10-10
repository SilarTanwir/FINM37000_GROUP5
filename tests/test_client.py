"""Tests for data/client.py (PLANNING.md §5.4, roadmap 3). No network, no API key."""

import logging

import pandas as pd
import pytest

from esvrp.data.client import (
    API_KEY_VAR,
    BudgetExceededError,
    MissingApiKeyError,
    cache_key,
    fetch,
    get_api_key,
    request_params,
)

ARGS = dict(dataset="GLBX.MDP3", schema="definition", symbols=["EW.OPT"],
            start="2026-09-01", end="2026-09-02", stype_in="parent")


class FakeStore:
    def __init__(self, df):
        self._df = df

    def to_df(self):
        return self._df


class FakeClient:
    """Stands in for databento.Historical; records every call in order."""

    def __init__(self, cost=1.0):
        self.calls = []
        self.cost = cost
        client = self

        class Metadata:
            def get_cost(self, **kw):
                client.calls.append(("get_cost", kw))
                return client.cost

        class Timeseries:
            def get_range(self, **kw):
                client.calls.append(("get_range", kw))
                return FakeStore(pd.DataFrame({"strike_price": [5000.0, 5100.0]}))

        self.metadata = Metadata()
        self.timeseries = Timeseries()


def test_cost_logged_before_pull(tmp_path, caplog):
    client = FakeClient(cost=2.5)
    with caplog.at_level(logging.INFO, logger="esvrp.data.client"):
        df = fetch(**ARGS, budget_usd=10, cache_dir=tmp_path, client=client)
    assert [name for name, _ in client.calls] == ["get_cost", "get_range"]
    assert "$2.50" in caplog.text
    assert len(df) == 2


def test_budget_abort_does_not_pull(tmp_path):
    client = FakeClient(cost=99.0)
    with pytest.raises(BudgetExceededError):
        fetch(**ARGS, budget_usd=10, cache_dir=tmp_path, client=client)
    assert [name for name, _ in client.calls] == ["get_cost"]
    assert not list(tmp_path.iterdir())


def test_repeated_call_hits_cache(tmp_path):
    first = fetch(**ARGS, budget_usd=10, cache_dir=tmp_path, client=FakeClient())
    client = FakeClient()
    second = fetch(**ARGS, budget_usd=10, cache_dir=tmp_path, client=client)
    assert client.calls == []
    pd.testing.assert_frame_equal(first, second)


def test_cache_writes_request_sidecar(tmp_path):
    fetch(**ARGS, budget_usd=10, cache_dir=tmp_path, client=FakeClient(cost=1.25))
    sidecars = list(tmp_path.glob("*.json"))
    assert len(sidecars) == 1
    meta = pd.read_json(sidecars[0], typ="series")
    assert meta["schema"] == "definition" and meta["estimated_cost_usd"] == 1.25


def test_cache_key_ignores_symbol_order_but_not_params():
    a = cache_key(request_params("GLBX.MDP3", "bbo-1m", ["B", "A"], "2026-09-01", "2026-09-02"))
    b = cache_key(request_params("GLBX.MDP3", "bbo-1m", ["A", "B"], "2026-09-01", "2026-09-02"))
    c = cache_key(request_params("GLBX.MDP3", "bbo-1m", ["A", "B"], "2026-09-01", "2026-09-03"))
    assert a == b != c


def test_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv(API_KEY_VAR, raising=False)
    with pytest.raises(MissingApiKeyError):
        get_api_key(env_path=tmp_path / ".env")


def test_key_read_from_env_file_and_never_logged(monkeypatch, tmp_path, caplog):
    monkeypatch.delenv(API_KEY_VAR, raising=False)
    env = tmp_path / ".env"
    env.write_text(f"{API_KEY_VAR}=db-secret-123\n")
    with caplog.at_level(logging.DEBUG):
        assert get_api_key(env_path=env) == "db-secret-123"
        fetch(**ARGS, budget_usd=10, cache_dir=tmp_path, client=FakeClient())
    assert "db-secret-123" not in caplog.text
    monkeypatch.delenv(API_KEY_VAR, raising=False)


def test_transient_timeout_is_retried(tmp_path, monkeypatch):
    import requests

    import esvrp.data.client as client_mod

    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    client = FakeClient()
    real_get_cost = client.metadata.get_cost
    failures = iter([True, False])

    def flaky_get_cost(**kw):
        if next(failures):
            raise requests.Timeout("read timed out")
        return real_get_cost(**kw)

    client.metadata.get_cost = flaky_get_cost
    fetch(**ARGS, budget_usd=10, cache_dir=tmp_path, client=client)
    assert [name for name, _ in client.calls] == ["get_cost", "get_range"]
