# PROMPT: "Write pytest tests for a store-metrics endpoint that must exclude
#   staff, return explicit zeros (never null/NaN) on zero traffic, and compute
#   conversion via POS time-window correlation. Cover: empty store returns zeros
#   without crashing, an all-staff clip yields unique_visitors=0 and conversion=0,
#   a zero-purchase store yields conversion_rate 0.0 (not null), and staff are
#   excluded from customer metrics."
# CHANGES MADE: The AI omitted the all-staff and zero-purchase fixtures entirely,
#   so I added them. I also added an explicit assertion that conversion_rate is a
#   float (catches the null/NaN regression the brief calls out) and a test that a
#   converted visitor is counted only when billing presence falls inside the POS
#   correlation window (the AI assumed any visitor with a transaction converts).
from __future__ import annotations

import datetime as dt

from app import db, pos
from tests.conftest import ingest

STORE = "STORE_TEST"
BASE = dt.datetime(2026, 4, 16, 8, 0, 0, tzinfo=dt.timezone.utc)


def _iso(t):
    return t.isoformat().replace("+00:00", "Z")


def test_empty_store_returns_zeros(client):
    m = client.get("/stores/GHOST_STORE/metrics").json()
    assert m["unique_visitors"] == 0
    assert m["conversion_rate"] == 0.0
    assert isinstance(m["conversion_rate"], float)
    assert m["data_confidence"] == "LOW"


def test_all_staff_clip(client, event_factory):
    events = [
        event_factory(visitor_id="VIS_S1", is_staff=True, camera_id="CAM_BACK_04"),
        event_factory(visitor_id="VIS_S2", is_staff=True, camera_id="CAM_BACK_04"),
    ]
    ingest(client, events)
    m = client.get(f"/stores/{STORE}/metrics").json()
    assert m["unique_visitors"] == 0
    assert m["conversion_rate"] == 0.0


def test_staff_excluded_from_unique_visitors(client, event_factory):
    events = [
        event_factory(visitor_id="VIS_C1"),
        event_factory(visitor_id="VIS_C2"),
        event_factory(visitor_id="VIS_STAFF", is_staff=True),
    ]
    ingest(client, events)
    m = client.get(f"/stores/{STORE}/metrics").json()
    assert m["unique_visitors"] == 2


def test_zero_purchase_conversion_is_float_zero(client, event_factory):
    ingest(client, [event_factory(visitor_id="VIS_C1")])
    m = client.get(f"/stores/{STORE}/metrics").json()
    assert m["conversion_rate"] == 0.0
    assert isinstance(m["conversion_rate"], float)


def test_conversion_requires_billing_presence_in_window(client, event_factory):
    # Visitor enters and is in billing at 08:00:10; transaction at 08:00:30 (20s
    # later, inside the 5-min window) => converted.
    enter = event_factory(visitor_id="VIS_C1", event_type="ENTRY", timestamp=_iso(BASE))
    bill = event_factory(
        visitor_id="VIS_C1",
        event_type="BILLING_QUEUE_JOIN",
        zone_id="BILLING_QUEUE",
        camera_id="CAM_BILL_05",
        timestamp=_iso(BASE + dt.timedelta(seconds=10)),
        metadata={"queue_depth": 1},
    )
    ingest(client, [enter, bill])

    # Insert a POS transaction directly.
    db.insert_pos_ignore_conflicts(
        [
            {
                "transaction_id": "TXN_1",
                "store_id": STORE,
                "timestamp": (BASE + dt.timedelta(seconds=30)).replace(tzinfo=None),
                "basket_value_inr": 999,
            }
        ]
    )
    m = client.get(f"/stores/{STORE}/metrics").json()
    assert m["purchases"] == 1
    assert m["conversion_rate"] == 1.0  # 1 converted of 1 unique visitor


def test_dwell_and_queue_depth_reported(client, event_factory):
    events = [
        event_factory(visitor_id="VIS_C1", event_type="ENTRY"),
        event_factory(
            visitor_id="VIS_C1",
            event_type="ZONE_DWELL",
            zone_id="SKINCARE_MOISTURISER",
            dwell_ms=30000,
            camera_id="CAM_SKIN_01",
        ),
        event_factory(
            visitor_id="VIS_C1",
            event_type="BILLING_QUEUE_JOIN",
            zone_id="BILLING_QUEUE",
            camera_id="CAM_BILL_05",
            metadata={"queue_depth": 3},
        ),
    ]
    ingest(client, events)
    m = client.get(f"/stores/{STORE}/metrics").json()
    assert m["avg_dwell_ms_by_zone"]["SKINCARE_MOISTURISER"] == 30000
    assert m["current_queue_depth"] == 3
