"""Unit tests for the resilience layer (circuit breaker + snapshot cache).
No network calls; matches the style of tests/test_core.py.
"""
import json
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from fpl_copilot.resilience import (
    CircuitBreaker, CircuitOpenError, SnapshotCache, FplApiDegraded,
    resilient_call,
)


def failing():
    raise RuntimeError("upstream down")


def working():
    return {"ok": True}


# ---------------------------------------------------------- circuit breaker --
def test_circuit_stays_closed_below_threshold():
    cb = CircuitBreaker(failure_threshold=4, reset_after=300)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            cb.call(failing)
    assert cb.state == "closed"


def test_circuit_opens_at_threshold():
    cb = CircuitBreaker(failure_threshold=4, reset_after=300)
    for _ in range(4):
        with pytest.raises(RuntimeError):
            cb.call(failing)
    assert cb.state == "open"
    with pytest.raises(CircuitOpenError):
        cb.call(failing)


def test_circuit_closes_on_success():
    cb = CircuitBreaker(failure_threshold=2, reset_after=300)
    with pytest.raises(RuntimeError):
        cb.call(failing)
    assert cb.call(working) == {"ok": True}
    assert cb.state == "closed"


def test_circuit_half_open_after_cooldown():
    cb = CircuitBreaker(failure_threshold=1, reset_after=0.01)
    with pytest.raises(RuntimeError):
        cb.call(failing)
    assert cb.state == "open"
    time.sleep(0.02)
    assert cb.state == "half-open"
    assert cb.call(working) == {"ok": True}
    assert cb.state == "closed"


def test_circuit_reopens_if_half_open_probe_fails():
    cb = CircuitBreaker(failure_threshold=1, reset_after=0.01)
    with pytest.raises(RuntimeError):
        cb.call(failing)
    time.sleep(0.02)
    assert cb.state == "half-open"
    with pytest.raises(RuntimeError):
        cb.call(failing)
    assert cb.state == "open"


# -------------------------------------------------------------- snapshot cache --
def test_snapshot_cache_returns_most_recent(tmp_path):
    (tmp_path / "gw01").mkdir()
    (tmp_path / "gw02").mkdir()
    (tmp_path / "gw01" / "bootstrap.json").write_text(json.dumps({"gw": 1}))
    (tmp_path / "gw02" / "bootstrap.json").write_text(json.dumps({"gw": 2}))
    cache = SnapshotCache(tmp_path)
    data, source = cache.most_recent("bootstrap")
    assert data == {"gw": 2}
    assert source.name == "bootstrap.json"


def test_snapshot_cache_empty_returns_none(tmp_path):
    cache = SnapshotCache(tmp_path)
    data, source = cache.most_recent("bootstrap")
    assert data is None and source is None


def test_snapshot_cache_missing_root_returns_none(tmp_path):
    cache = SnapshotCache(tmp_path / "does-not-exist")
    data, source = cache.most_recent("bootstrap")
    assert data is None and source is None


# ------------------------------------------------------------------ resilient_call --
def test_resilient_call_degrades_when_cache_available(tmp_path):
    (tmp_path / "gw01").mkdir()
    (tmp_path / "gw01" / "bootstrap.json").write_text(json.dumps({"gw": 1}))
    cache = SnapshotCache(tmp_path)
    cb = CircuitBreaker(failure_threshold=1, reset_after=300)
    with pytest.raises(FplApiDegraded) as excinfo:
        resilient_call(cb, failing, cache, "bootstrap")
    assert excinfo.value.data == {"gw": 1}


def test_resilient_call_raises_when_no_cache(tmp_path):
    cache = SnapshotCache(tmp_path)
    cb = CircuitBreaker(failure_threshold=1, reset_after=300)
    with pytest.raises(RuntimeError):
        resilient_call(cb, failing, cache, "bootstrap")


def test_resilient_call_succeeds_normally():
    cache = SnapshotCache(pathlib.Path("/nonexistent"))
    cb = CircuitBreaker(failure_threshold=4, reset_after=300)
    assert resilient_call(cb, working, cache, "bootstrap") == {"ok": True}
