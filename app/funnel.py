"""Entry-anchored aggregate funnel (BUILD_SPEC Section 10.3).

Stages: ENTRY -> ZONE_VISIT -> BILLING_QUEUE -> PURCHASE.

The unit is the visitor, **deduped on visitor_id** so a re-entry (same token)
never double-counts. This is an *aggregate* funnel (unique counts per stage),
not a per-individual trace -- because reliable cross-camera linking is not
achievable on this footage. That honesty is documented in DESIGN.md.
"""
from __future__ import annotations

from typing import Any, Optional

from app import db, pos
from app.metrics import resolve_window, visitor_base


def compute_funnel(store_id: str, window_min: Optional[int] = None) -> dict[str, Any]:
    start, end = resolve_window(store_id, window_min)
    events = db.fetch_events(store_id, start, end, include_staff=False)

    # Entry-anchored when entries exist; else floor-derived (mid-session clip with
    # no door crossings) so the funnel head isn't an empty 0 on a busy store.
    entered, basis = visitor_base(events)

    zone_visitors = {
        e["visitor_id"]
        for e in events
        if e["event_type"] in {"ZONE_ENTER", "ZONE_DWELL"}
    }
    # Stage 2 is a subset relationship at the aggregate level: visitors who both
    # entered and visited a zone (where the link is known), else any zone visitor.
    stage2 = (entered & zone_visitors) or zone_visitors

    # Reaching the billing queue = a JOIN or an ABANDON (both prove queue entry;
    # the new schema emits one or the other, legacy emits both).
    queue_visitors = {
        e["visitor_id"]
        for e in events
        if e["event_type"] in {"BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON"}
    }
    stage3 = (stage2 & queue_visitors) or queue_visitors

    conv = pos.correlate_conversions(store_id, start, end)
    purchase_count = conv["purchase_count"]

    counts = [
        ("ENTRY", len(entered)),
        ("ZONE_VISIT", len(stage2)),
        ("BILLING_QUEUE", len(stage3)),
        ("PURCHASE", purchase_count),
    ]

    stages = []
    prev = None
    for name, count in counts:
        if prev is None or prev == 0:
            drop = 0.0
        else:
            drop = round(max(0.0, (prev - count) / prev) * 100, 2)
        stages.append({"stage": name, "count": count, "drop_off_pct": drop})
        prev = count

    note = "aggregate unique-count funnel; not a per-individual trace"
    if basis == "floor":
        note += " (ENTRY stage is floor-derived: no door crossings in this window)"

    return {
        "store_id": store_id,
        "stages": stages,
        "visitor_basis": basis,
        "note": note,
        "window": {
            "start": start.isoformat().replace("+00:00", "Z") if start else None,
            "end": end.isoformat().replace("+00:00", "Z") if end else None,
            "window_min": window_min,
        },
    }
