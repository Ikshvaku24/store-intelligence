# PROMPT: "Write pytest tests for a /health endpoint that reports DB connectivity
#   and per-store last-event lag with a stale_feed flag, and for the graceful
#   degradation path where a DB OperationalError becomes an HTTP 503 with a
#   structured body (no stack trace)."
# CHANGES MADE: The AI tested only the happy path. I added the 503 degradation
#   test by monkeypatching the metrics layer to raise OperationalError and
#   asserting the middleware converts it to a clean 503 body with a trace_id and
#   no traceback. Also asserted stale_feed=True for the historical sample clip
#   (its timestamps are well past the staleness threshold).
from __future__ import annotations

import pytest
from sqlalchemy.exc import OperationalError

from tests.conftest import ingest


def test_health_ok_empty(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["db"] == "ok"
    assert h["stores"] == []
    assert "version" in h


def test_health_reports_stale_feed(client, sample_events):
    ingest(client, sample_events)
    h = client.get("/health").json()
    assert h["stores"], "expected at least one store after ingest"
    store = h["stores"][0]  # the sample is a single historical store clip
    assert store["stale_feed"] is True  # sample clip is from the past
    assert store["lag_seconds"] > 0


def test_db_down_returns_503(client, monkeypatch):
    def boom(*a, **k):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr("app.main.metrics.compute_metrics", boom)
    resp = client.get("/stores/ANY/metrics")
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "database_unavailable"
    assert "trace_id" in body
    # No stack trace leaked.
    assert "Traceback" not in str(body)


def test_root_endpoint(client):
    r = client.get("/").json()
    assert r["service"] == "store-intelligence"
