"""Shared test fixtures (BUILD_SPEC Section 14.1).

* An ephemeral SQLite database per test (no server needed; the same Core code
  path -- dialect-aware on_conflict_do_nothing -- runs against Postgres in CI/docker).
* A StoreEvent factory minting valid events with overridable fields.
* A loader for the sample_events.jsonl fixture.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "sample_events.jsonl"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient backed by a fresh SQLite file; tables created on startup."""
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_file.as_posix()}")
    monkeypatch.setenv("STORE_LAYOUT_PATH", str(ROOT / "data" / "store_layout.json"))
    monkeypatch.setenv("POS_CSV_PATH", "")  # don't auto-load POS unless a test asks

    # Rebuild cached settings/engine so the new DB_URL takes effect.
    from app import db
    from app.config import get_settings

    get_settings.cache_clear()
    db.reset_engine()

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c

    db.reset_engine()
    get_settings.cache_clear()


@pytest.fixture()
def event_factory():
    base = dt.datetime(2026, 4, 16, 8, 0, 0, tzinfo=dt.timezone.utc)
    counter = {"n": 0}

    def make(**overrides: Any) -> dict[str, Any]:
        counter["n"] += 1
        ev = {
            "event_id": overrides.pop("event_id", str(uuid.uuid4())),
            "store_id": "STORE_TEST",
            "camera_id": "CAM_ENTRY_03",
            "visitor_id": f"VIS_{counter['n']:04d}",
            "event_type": "ENTRY",
            "timestamp": (base + dt.timedelta(seconds=counter["n"])).isoformat().replace("+00:00", "Z"),
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.8,
            "metadata": {},
        }
        ev.update(overrides)
        return ev

    return make


@pytest.fixture()
def sample_events() -> list[dict[str, Any]]:
    with open(SAMPLE_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def ingest(client, events: list[dict[str, Any]]):
    return client.post("/events/ingest", json=events)
