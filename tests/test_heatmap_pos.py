# PROMPT: "Write pytest tests for the heatmap endpoint (per-zone visit_count,
#   avg_dwell_ms, 0-100 score, data_confidence LOW below the session threshold)
#   and for POS loading from CSV plus the layout open-hours helper."
# CHANGES MADE: Added the explicit normalisation check (the busiest zone scores
#   100) which the AI left out, and a closed-hours case for the layout helper so
#   the DEAD_ZONE 'only during open hours' rule is actually exercised. Used the
#   real load_pos_csv path against a temp CSV rather than mocking the reader.
from __future__ import annotations

import datetime as dt

from app import layout, pos
from app.config import get_settings
from tests.conftest import ingest

STORE = "STORE_TEST"


def test_heatmap_scores_and_confidence(client, event_factory):
    events = [event_factory(visitor_id=f"V{i}", event_type="ENTRY") for i in range(3)]
    # Zone A visited by 3, zone B by 1.
    for i in range(3):
        events.append(event_factory(visitor_id=f"V{i}", event_type="ZONE_ENTER", zone_id="MAKEUP_LIPS", camera_id="CAM_MAKEUP_02"))
        events.append(event_factory(visitor_id=f"V{i}", event_type="ZONE_EXIT", zone_id="MAKEUP_LIPS", camera_id="CAM_MAKEUP_02", dwell_ms=20000))
    events.append(event_factory(visitor_id="V0", event_type="ZONE_ENTER", zone_id="SKINCARE_CLEANSER", camera_id="CAM_SKIN_01"))
    ingest(client, events)

    h = client.get(f"/stores/{STORE}/heatmap").json()
    zones = {z["zone_id"]: z for z in h["zones"]}
    assert zones["MAKEUP_LIPS"]["visit_count"] == 3
    assert zones["MAKEUP_LIPS"]["score_0_100"] == 100.0  # busiest -> 100
    assert zones["SKINCARE_CLEANSER"]["score_0_100"] < 100.0
    assert zones["MAKEUP_LIPS"]["avg_dwell_ms"] == 20000
    # Only 3 sessions < default threshold -> LOW.
    assert h["data_confidence"] == "LOW"


def test_heatmap_empty_store(client):
    h = client.get("/stores/NOPE/heatmap").json()
    assert h["zones"] == []
    assert h["data_confidence"] == "LOW"


def test_load_pos_csv_and_correlate(client, tmp_path, event_factory):
    base = dt.datetime(2026, 4, 16, 8, 0, 0, tzinfo=dt.timezone.utc)
    # Customer in billing zone, then a transaction 30s later.
    events = [
        event_factory(visitor_id="VC", event_type="ENTRY", timestamp=base.isoformat().replace("+00:00", "Z")),
        event_factory(
            visitor_id="VC",
            event_type="BILLING_QUEUE_JOIN",
            zone_id="BILLING_QUEUE",
            camera_id="CAM_BILL_05",
            timestamp=(base + dt.timedelta(seconds=10)).isoformat().replace("+00:00", "Z"),
            metadata={"queue_depth": 1},
        ),
    ]
    ingest(client, events)

    csv_path = tmp_path / "pos.csv"
    csv_path.write_text(
        "store_id,transaction_id,timestamp,basket_value_inr\n"
        f"{STORE},TXN_A,{(base + dt.timedelta(seconds=30)).isoformat().replace('+00:00','Z')},999\n",
        encoding="utf-8",
    )
    rows = pos.load_pos_csv(str(csv_path))
    assert rows == 1

    # Re-loading is idempotent (conflict-ignore on transaction_id).
    assert pos.load_pos_csv(str(csv_path)) == 1

    m = client.get(f"/stores/{STORE}/metrics").json()
    assert m["purchases"] == 1
    assert m["conversion_rate"] == 1.0


def test_layout_open_hours(monkeypatch, tmp_path):
    layout_file = tmp_path / "layout.json"
    layout_file.write_text(
        '{"stores": {"S1": {"store_id": "S1", "open_hours": {"open": "09:00", "close": "21:00"}}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("STORE_LAYOUT_PATH", str(layout_file))
    get_settings.cache_clear()
    layout.load_layout.cache_clear()

    open_t = dt.datetime(2026, 4, 16, 12, 0, tzinfo=dt.timezone.utc)
    closed_t = dt.datetime(2026, 4, 16, 23, 0, tzinfo=dt.timezone.utc)
    assert layout.is_open_at("S1", open_t) is True
    assert layout.is_open_at("S1", closed_t) is False
    # Unknown store -> always open (never suppress a genuine signal).
    assert layout.is_open_at("UNKNOWN", closed_t) is True

    layout.load_layout.cache_clear()
    get_settings.cache_clear()
