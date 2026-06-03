# PROMPT: "Write pytest tests for the new-schema ingestion adapter: the graders'
#   sample_events.jsonl (entry/zone/queue shapes with id_token vs track_id, and
#   store_code vs store_id) must ingest end-to-end through the real API, fold to a
#   single canonical store, drive non-zero metrics/funnel, and stay idempotent.
#   Also unit-test store-id canonicalisation and per-camera track-id namespacing."
# CHANGES MADE: Added the back-compat assertion that an already-canonical event
#   still passes through unchanged, and the partial-success case where a genuinely
#   malformed event is rejected without sinking a valid new-schema batch.
from __future__ import annotations

from app.normalize import canonical_store_id, to_canonical
from tests.conftest import ingest


def test_store_id_folding():
    assert canonical_store_id("store_1076") == "ST1076"
    assert canonical_store_id("STORE_1076") == "ST1076"
    assert canonical_store_id("ST1076") == "ST1076"
    assert canonical_store_id("STORE_BLR_002") == "STORE_BLR_002"  # left intact
    assert canonical_store_id(None) is None


def test_entry_event_maps_to_canonical():
    raw = {
        "event_type": "entry", "id_token": "ID_1", "store_code": "store_1076",
        "camera_id": "cam1", "event_timestamp": "2026-03-08T18:10:05.120000",
        "is_staff": False, "gender_pred": "F", "age_pred": 28, "group_id": "G_10",
        "group_size": 2,
    }
    rows = to_canonical(raw)
    assert len(rows) == 1
    r = rows[0]
    assert r["event_type"] == "ENTRY"
    assert r["store_id"] == "ST1076"
    assert r["visitor_id"] == "ID_1"
    assert r["metadata"]["gender"] == "F" and r["metadata"]["group_size"] == 2


def test_zone_track_id_is_namespaced_per_camera():
    base = {
        "event_type": "zone_entered", "track_id": 101, "store_id": "ST1076",
        "zone_id": "Z01", "event_time": "2026-03-08T18:10:45.280000",
    }
    a = to_canonical({**base, "camera_id": "CAM2"})[0]
    b = to_canonical({**base, "camera_id": "CAM3"})[0]
    # same integer track on two cameras must NOT collapse into one visitor
    assert a["visitor_id"] == "T101@CAM2"
    assert b["visitor_id"] == "T101@CAM3"
    assert a["visitor_id"] != b["visitor_id"]


def test_canonical_event_passes_through(event_factory):
    raw = event_factory()  # already-canonical ENTRY event
    assert to_canonical(raw) == [raw]


def test_provided_sample_ingests_and_drives_metrics(client, sample_events):
    r = ingest(client, sample_events).json()
    assert r["rejected"] == 0
    assert r["accepted"] == len(sample_events)  # 1:1 mapping for this sample

    # entry/zone/queue all folded under the one canonical store id
    m = client.get("/stores/ST1076/metrics").json()
    assert m["unique_visitors"] == 3            # three entry id_tokens
    assert m["abandonment_rate"] > 0            # one queue_abandoned of the queue base

    # P2: demographics + groups attributed over the entry-visitor base (not per
    # camera token), and zone dwell derived by pairing enter/exit (no dwell_ms).
    assert m["demographics"]["by_gender"] == {"F": 2, "M": 1}
    assert m["groups"]["group_count"] == 1 and m["groups"]["visitors_in_groups"] == 2
    assert m["avg_dwell_ms_by_zone"].get("PURPLLE_MUM_1076_Z01", 0) > 0
    assert m["avg_queue_wait_seconds"] > 0

    f = client.get("/stores/ST1076/funnel").json()
    stages = {s["stage"]: s["count"] for s in f["stages"]}
    assert stages["ENTRY"] == 3
    assert stages["BILLING_QUEUE"] >= 1


def test_new_schema_is_idempotent(client, sample_events):
    ingest(client, sample_events)
    r2 = ingest(client, sample_events).json()
    assert r2["accepted"] == 0
    assert r2["duplicates"] == len(sample_events)


def test_partial_success_mixed_batch(client, sample_events):
    bad = {"event_type": "entry"}  # missing id_token + timestamp -> rejected
    r = ingest(client, sample_events + [bad]).json()
    assert r["accepted"] == len(sample_events)
    assert r["rejected"] == 1
