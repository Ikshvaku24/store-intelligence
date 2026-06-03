# PROMPT: "Write pytest tests for the visitor-counting fallback: when a clip
#   window has ENTRY events, unique_visitors is entry-anchored (basis 'entry');
#   when it has NO entries but non-staff floor activity (the real footage where
#   nobody crosses the door but the store is full), unique_visitors falls back to
#   distinct floor visitor_ids (basis 'floor', data_confidence LOW). Cover metrics
#   and funnel, and confirm the conversion denominator uses the fallback base."
# CHANGES MADE: The AI's first version asserted data_confidence 'OK' for the floor
#   path; I corrected it to 'LOW' because the floor count is per-camera and
#   cross-camera dedup is still best-effort, so it must be flagged. Added the
#   funnel ENTRY-stage assertion (the head must equal the floor base, not 0) and a
#   guard that the entry-anchored path is unchanged for graded data.
from __future__ import annotations

from tests.conftest import ingest

STORE = "STORE_TEST"


def _zone(event_factory, vid, etype="ZONE_ENTER", zone="MAKEUP_LIPS", **kw):
    return event_factory(
        visitor_id=vid, event_type=etype, zone_id=zone, camera_id="CAM_MAKEUP_02", **kw
    )


def test_metrics_entry_basis_when_entries_present(client, event_factory):
    ingest(client, [event_factory(visitor_id="V1", event_type="ENTRY")])
    m = client.get(f"/stores/{STORE}/metrics").json()
    assert m["visitor_basis"] == "entry"
    assert m["unique_visitors"] == 1
    assert m["data_confidence"] == "OK"


def test_metrics_floor_fallback_when_no_entries(client, event_factory):
    events = [
        _zone(event_factory, "VF1"),
        _zone(event_factory, "VF2"),
        _zone(event_factory, "VF2", etype="ZONE_EXIT", dwell_ms=4000),
    ]
    ingest(client, events)
    m = client.get(f"/stores/{STORE}/metrics").json()
    assert m["visitor_basis"] == "floor"
    assert m["unique_visitors"] == 2          # distinct floor tokens, not 0
    assert m["data_confidence"] == "LOW"      # approximate -> flagged
    assert isinstance(m["conversion_rate"], float)


def test_floor_basis_excludes_staff(client, event_factory):
    events = [
        _zone(event_factory, "VF1"),
        _zone(event_factory, "VSTAFF", is_staff=True),
    ]
    ingest(client, events)
    m = client.get(f"/stores/{STORE}/metrics").json()
    assert m["unique_visitors"] == 1          # staff excluded from the floor base


def test_funnel_floor_fallback_head_not_zero(client, event_factory):
    events = [_zone(event_factory, "VF1"), _zone(event_factory, "VF2")]
    ingest(client, events)
    f = client.get(f"/stores/{STORE}/funnel").json()
    assert f["visitor_basis"] == "floor"
    counts = {s["stage"]: s["count"] for s in f["stages"]}
    assert counts["ENTRY"] == 2               # floor-derived head, not an empty 0
    assert counts["ZONE_VISIT"] == 2


def test_empty_store_still_zero(client):
    m = client.get("/stores/GHOST/metrics").json()
    assert m["unique_visitors"] == 0
    assert m["conversion_rate"] == 0.0
    assert m["data_confidence"] == "LOW"
