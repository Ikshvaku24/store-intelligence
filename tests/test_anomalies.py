# PROMPT: "Write pytest tests for an anomalies endpoint with three detectors:
#   BILLING_QUEUE_SPIKE (from queue depth), CONVERSION_DROP (returns
#   baseline='insufficient_history' on a short window), and DEAD_ZONE (no activity
#   for a configurable window during open hours). Assert each result carries a
#   severity in {INFO,WARN,CRITICAL} and a suggested_action."
# CHANGES MADE: The AI's CONVERSION_DROP test expected a fabricated trend on a
#   90-second window; I rewrote it to assert the honest insufficient_history path
#   instead. Added the DEAD_ZONE test using a real two-event gap longer than the
#   configured window (the AI mocked time, which hid a timezone bug). Asserted the
#   structured fields (type/severity/detected_at/suggested_action) exist on every
#   anomaly so the response contract is enforced.
from __future__ import annotations

import datetime as dt

from tests.conftest import ingest

STORE = "STORE_TEST"
BASE = dt.datetime(2026, 4, 16, 8, 0, 0, tzinfo=dt.timezone.utc)


def _iso(t):
    return t.isoformat().replace("+00:00", "Z")


def _types(anoms):
    return {a["type"] for a in anoms["anomalies"]}


def test_queue_spike_fires(client, event_factory):
    events = [
        event_factory(visitor_id="V0", event_type="ENTRY"),
        event_factory(
            visitor_id="V0",
            event_type="BILLING_QUEUE_JOIN",
            zone_id="BILLING_QUEUE",
            camera_id="CAM_BILL_05",
            metadata={"queue_depth": 6},
        ),
    ]
    ingest(client, events)
    a = client.get(f"/stores/{STORE}/anomalies").json()
    assert "BILLING_QUEUE_SPIKE" in _types(a)
    spike = next(x for x in a["anomalies"] if x["type"] == "BILLING_QUEUE_SPIKE")
    assert spike["severity"] in {"WARN", "CRITICAL"}
    assert spike["suggested_action"]


def test_no_spike_below_threshold(client, event_factory):
    events = [
        event_factory(
            visitor_id="V0",
            event_type="BILLING_QUEUE_JOIN",
            zone_id="BILLING_QUEUE",
            camera_id="CAM_BILL_05",
            metadata={"queue_depth": 2},
        ),
    ]
    ingest(client, events)
    a = client.get(f"/stores/{STORE}/anomalies").json()
    assert "BILLING_QUEUE_SPIKE" not in _types(a)


def test_conversion_drop_insufficient_history(client, event_factory):
    # Short window (seconds) => insufficient_history, severity INFO, no fabrication.
    events = [
        event_factory(visitor_id="V0", event_type="ENTRY", timestamp=_iso(BASE)),
        event_factory(visitor_id="V1", event_type="ENTRY", timestamp=_iso(BASE + dt.timedelta(seconds=30))),
    ]
    ingest(client, events)
    a = client.get(f"/stores/{STORE}/anomalies").json()
    drop = next((x for x in a["anomalies"] if x["type"] == "CONVERSION_DROP"), None)
    assert drop is not None
    assert drop["baseline"] == "insufficient_history"
    assert drop["severity"] == "INFO"


def test_dead_zone_fires_on_gap(client, event_factory):
    # Two activity events 12 minutes apart -> a >5min gap during open hours.
    events = [
        event_factory(visitor_id="V0", event_type="ENTRY", timestamp=_iso(BASE)),
        event_factory(visitor_id="V1", event_type="ENTRY", timestamp=_iso(BASE + dt.timedelta(minutes=12))),
    ]
    ingest(client, events)
    a = client.get(f"/stores/{STORE}/anomalies").json()
    assert "DEAD_ZONE" in _types(a)
    dz = next(x for x in a["anomalies"] if x["type"] == "DEAD_ZONE")
    assert dz["severity"] in {"INFO", "WARN", "CRITICAL"}
    assert dz["suggested_action"]


def test_all_anomalies_have_required_fields(client, sample_events):
    ingest(client, sample_events)
    a = client.get("/stores/ST1076/anomalies").json()
    for anom in a["anomalies"]:
        assert anom["type"]
        assert anom["severity"] in {"INFO", "WARN", "CRITICAL"}
        assert "detected_at" in anom
        assert anom["suggested_action"]
