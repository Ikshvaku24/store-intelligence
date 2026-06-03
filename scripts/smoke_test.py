"""End-to-end smoke test against the in-process app using SQLite.

Run: DB_URL=sqlite:///./smoke.db python scripts/smoke_test.py
Exercises every endpoint + idempotency without needing Docker/Postgres.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("DB_URL", "sqlite:///./smoke.db")
os.environ.setdefault("STORE_LAYOUT_PATH", "data/store_layout.json")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

STORE = "STORE_BLR_002"


def load_events():
    with open("data/sample_events.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    with TestClient(app) as client:
        _run(client)


def _run(client):
    events = load_events()

    # health (empty)
    print("health:", client.get("/health").json())

    # ingest
    r1 = client.post("/events/ingest", json=events).json()
    print("ingest #1:", {k: r1[k] for k in ("received", "accepted", "duplicates", "rejected")})

    # idempotency: re-POST -> accepted 0, duplicates N
    r2 = client.post("/events/ingest", json=events).json()
    print("ingest #2 (idempotent):", {k: r2[k] for k in ("received", "accepted", "duplicates", "rejected")})
    assert r2["accepted"] == 0 and r2["duplicates"] == r1["received"] - r1["rejected"], "idempotency broken"

    # malformed partial success
    bad = [{"event_id": "x", "bogus": True}]
    rb = client.post("/events/ingest", json=bad).json()
    print("malformed:", {k: rb[k] for k in ("received", "accepted", "rejected")})
    assert rb["rejected"] == 1

    # metrics / funnel / heatmap / anomalies
    for ep in ("metrics", "funnel", "heatmap", "anomalies"):
        resp = client.get(f"/stores/{STORE}/{ep}")
        print(f"{ep}:", json.dumps(resp.json(), indent=None)[:400])
        assert resp.status_code == 200

    # empty store -> zeros not crash
    m_empty = client.get("/stores/NO_SUCH_STORE/metrics").json()
    print("empty metrics:", m_empty["unique_visitors"], m_empty["conversion_rate"], m_empty["data_confidence"])
    assert m_empty["unique_visitors"] == 0 and m_empty["conversion_rate"] == 0.0

    print("health (after):", client.get("/health").json())
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
