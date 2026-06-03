# PROMPT: "Write pytest tests for an entry-anchored aggregate funnel
#   (ENTRY -> ZONE_VISIT -> BILLING_QUEUE -> PURCHASE) whose unit is the visitor,
#   deduped on visitor_id so a re-entry never double-counts. Cover: a visitor who
#   exits and re-enters is counted once at ENTRY, aggregate stage counts with
#   drop-off percentages, and that low id_confidence links still aggregate."
# CHANGES MADE: The AI counted REENTRY events as new entries (double-counting the
#   re-entering visitor); I rewrote the ENTRY-stage assertion to prove the same
#   visitor_id collapses. Added the explicit drop_off_pct math check and a case
#   where a cross_camera_match (low id_confidence) zone visit still lands in the
#   aggregate stage count rather than being dropped.
from __future__ import annotations

from tests.conftest import ingest

STORE = "STORE_TEST"


def test_reentry_counted_once(client, event_factory):
    v = "VIS_RE"
    events = [
        event_factory(visitor_id=v, event_type="ENTRY"),
        event_factory(visitor_id=v, event_type="EXIT"),
        event_factory(
            visitor_id=v,
            event_type="REENTRY",
            metadata={"id_source": "reentry_match", "id_confidence": 0.66},
        ),
        event_factory(visitor_id=v, event_type="EXIT"),
        event_factory(visitor_id="VIS_OTHER", event_type="ENTRY"),
    ]
    ingest(client, events)
    f = client.get(f"/stores/{STORE}/funnel").json()
    entry_stage = next(s for s in f["stages"] if s["stage"] == "ENTRY")
    assert entry_stage["count"] == 2  # VIS_RE counted once + VIS_OTHER


def test_aggregate_stage_counts_and_dropoff(client, event_factory):
    # 4 entries; 3 visit a zone; 2 queue at billing.
    events = []
    for i in range(4):
        events.append(event_factory(visitor_id=f"V{i}", event_type="ENTRY"))
    for i in range(3):
        events.append(
            event_factory(visitor_id=f"V{i}", event_type="ZONE_ENTER", zone_id="MAKEUP_LIPS", camera_id="CAM_MAKEUP_02")
        )
    for i in range(2):
        events.append(
            event_factory(visitor_id=f"V{i}", event_type="BILLING_QUEUE_JOIN", zone_id="BILLING_QUEUE", camera_id="CAM_BILL_05", metadata={"queue_depth": 1})
        )
    ingest(client, events)
    f = client.get(f"/stores/{STORE}/funnel").json()
    counts = {s["stage"]: s["count"] for s in f["stages"]}
    assert counts["ENTRY"] == 4
    assert counts["ZONE_VISIT"] == 3
    assert counts["BILLING_QUEUE"] == 2
    # drop ENTRY->ZONE = (4-3)/4 = 25%
    zone_stage = next(s for s in f["stages"] if s["stage"] == "ZONE_VISIT")
    assert zone_stage["drop_off_pct"] == 25.0


def test_low_id_confidence_zone_still_aggregated(client, event_factory):
    events = [
        event_factory(visitor_id="V0", event_type="ENTRY"),
        event_factory(
            visitor_id="V0",
            event_type="ZONE_ENTER",
            zone_id="SKINCARE_CLEANSER",
            camera_id="CAM_SKIN_01",
            confidence=0.3,
            metadata={"id_source": "cross_camera_match", "id_confidence": 0.21},
        ),
    ]
    ingest(client, events)
    f = client.get(f"/stores/{STORE}/funnel").json()
    counts = {s["stage"]: s["count"] for s in f["stages"]}
    assert counts["ZONE_VISIT"] == 1  # not dropped despite low confidence


def test_staff_excluded_from_funnel(client, event_factory):
    events = [
        event_factory(visitor_id="VIS_C1", event_type="ENTRY"),
        event_factory(visitor_id="VIS_STAFF", event_type="ENTRY", is_staff=True),
    ]
    ingest(client, events)
    f = client.get(f"/stores/{STORE}/funnel").json()
    entry = next(s for s in f["stages"] if s["stage"] == "ENTRY")
    assert entry["count"] == 1
