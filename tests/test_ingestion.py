# PROMPT: "Write pytest tests for a FastAPI POST /events/ingest endpoint that
#   does batch validation with partial success and idempotent (conflict-ignore)
#   storage. Cover: double-POST yields identical DB state and metrics, a malformed
#   event is rejected individually without sinking the batch, batches over 500 are
#   rejected with a structured error, and an empty batch is handled."
# CHANGES MADE: The AI's first draft used an in-memory dict store, so it never
#   exercised the real ON CONFLICT path. Swapped it for the actual app on an
#   ephemeral SQLite DB (same dialect-aware upsert code as Postgres). Added the
#   "metrics identical after double POST" assertion (the AI only checked row
#   counts), and the {"events": [...]} envelope variant. Pinned the over-limit
#   status to 413 (Payload Too Large) -- the brief's "501" is a typo; 501 means
#   Not Implemented.
from __future__ import annotations

import uuid

from tests.conftest import ingest


def test_basic_ingest(client, sample_events):
    r = ingest(client, sample_events).json()
    assert r["received"] == len(sample_events)
    assert r["accepted"] == len(sample_events)
    assert r["duplicates"] == 0
    assert r["rejected"] == 0
    assert "trace_id" in r


def test_idempotent_double_post(client, sample_events):
    r1 = ingest(client, sample_events).json()
    metrics1 = client.get("/stores/ST1076/metrics").json()

    r2 = ingest(client, sample_events).json()
    metrics2 = client.get("/stores/ST1076/metrics").json()

    assert r2["accepted"] == 0
    assert r2["duplicates"] == r1["accepted"]
    # Identical state: every metric except the generated_at timestamp matches.
    for k in ("unique_visitors", "conversion_rate", "purchases", "abandonment_rate"):
        assert metrics1[k] == metrics2[k]


def test_partial_success_on_malformed(client, event_factory):
    good = event_factory()
    bad_missing = {"event_id": str(uuid.uuid4())}  # missing required fields
    bad_conf = event_factory(confidence=5.0)  # out of [0,1]
    r = ingest(client, [good, bad_missing, bad_conf]).json()
    assert r["received"] == 3
    assert r["accepted"] == 1
    assert r["rejected"] == 2
    assert len(r["errors"]) == 2
    assert all("index" in e and "reason" in e for e in r["errors"])


def test_within_batch_duplicate_counts_once(client, event_factory):
    eid = str(uuid.uuid4())
    e1 = event_factory(event_id=eid)
    e2 = event_factory(event_id=eid)  # same id twice in one batch
    r = ingest(client, [e1, e2]).json()
    assert r["accepted"] == 1
    assert r["duplicates"] == 1


def test_over_limit_rejected(client, event_factory):
    big = [event_factory() for _ in range(501)]
    resp = ingest(client, big)
    assert resp.status_code == 413
    body = resp.json()
    assert body["error"] == "batch_too_large"
    assert body["accepted"] == 0


def test_empty_batch(client):
    r = ingest(client, []).json()
    assert r["received"] == 0
    assert r["accepted"] == 0
    assert r["rejected"] == 0


def test_events_envelope_accepted(client, event_factory):
    resp = client.post("/events/ingest", json={"events": [event_factory()]})
    assert resp.status_code == 200
    assert resp.json()["accepted"] == 1


def test_trace_id_echoed_in_header(client, event_factory):
    resp = client.post("/events/ingest", json=[event_factory()], headers={"X-Trace-Id": "abc123"})
    assert resp.headers.get("X-Trace-Id") == "abc123"
    assert resp.json()["trace_id"] == "abc123"
