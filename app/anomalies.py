"""Anomaly detection (BUILD_SPEC Section 10.5).

Three honest detectors:
* BILLING_QUEUE_SPIKE -- real, computed from observed queue depth.
* CONVERSION_DROP     -- vs a sub-window baseline; returns
                         baseline="insufficient_history" (severity INFO) when the
                         window is too short to mean anything. Never fabricated.
* DEAD_ZONE           -- no customer activity for a configurable window, only
                         during open hours from store_layout.json.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from app import db, layout, pos
from app.config import get_settings
from app.metrics import now_utc, resolve_window

ACTIVITY_TYPES = {"ENTRY", "REENTRY", "ZONE_ENTER", "ZONE_DWELL"}


def compute_anomalies(store_id: str, window_min: Optional[int] = None) -> dict[str, Any]:
    start, end = resolve_window(store_id, window_min)
    events = db.fetch_events(store_id, start, end, include_staff=False)

    found: list[dict[str, Any]] = []
    found.extend(_queue_spike(events))
    drop = _conversion_drop(store_id, events, start, end)
    if drop:
        found.append(drop)
    found.extend(_dead_zone(store_id, events, start, end))

    return {
        "store_id": store_id,
        "anomalies": found,
        "window": {
            "start": start.isoformat().replace("+00:00", "Z") if start else None,
            "end": end.isoformat().replace("+00:00", "Z") if end else None,
            "window_min": window_min,
        },
    }


def _queue_spike(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settings = get_settings()
    out = []
    for e in events:
        if e["event_type"] != "BILLING_QUEUE_JOIN":
            continue
        depth = (e.get("metadata") or {}).get("queue_depth")
        if depth is None or depth < settings.queue_spike_threshold:
            continue
        severity = "CRITICAL" if depth >= settings.queue_spike_threshold + 2 else "WARN"
        out.append(
            {
                "type": "BILLING_QUEUE_SPIKE",
                "severity": severity,
                "detected_at": _iso(e["timestamp"]),
                "detail": f"Queue depth reached {depth} (threshold {settings.queue_spike_threshold}).",
                "suggested_action": "Open an additional billing counter or call floor staff to assist.",
            }
        )
    # Report only the worst spike to avoid noise.
    if not out:
        return []
    worst = max(out, key=lambda a: ("CRITICAL" == a["severity"], a["detail"]))
    return [worst]


def _conversion_drop(
    store_id: str,
    events: list[dict[str, Any]],
    start: Optional[dt.datetime],
    end: Optional[dt.datetime],
) -> Optional[dict[str, Any]]:
    settings = get_settings()
    if start is None or end is None:
        return None
    span_min = (end - start).total_seconds() / 60.0

    if span_min < settings.conversion_min_history_min:
        return {
            "type": "CONVERSION_DROP",
            "severity": "INFO",
            "detected_at": _iso(end),
            "detail": "Window too short to establish a conversion baseline.",
            "baseline": "insufficient_history",
            "suggested_action": "Collect at least "
            f"{settings.conversion_min_history_min} minutes of history before trending conversion.",
        }

    mid = start + (end - start) / 2
    first = _conversion_for(store_id, start, mid)
    second = _conversion_for(store_id, mid, end)
    if first <= 0:
        return None
    drop_frac = (first - second) / first
    if drop_frac < settings.conversion_drop_pct:
        return None
    return {
        "type": "CONVERSION_DROP",
        "severity": "WARN" if drop_frac < 0.5 else "CRITICAL",
        "detected_at": _iso(end),
        "detail": f"Conversion fell from {first:.2f} to {second:.2f} ({drop_frac*100:.0f}% drop).",
        "baseline": round(first, 4),
        "suggested_action": "Check staffing, queue length, and stock on high-traffic zones.",
    }


def _conversion_for(store_id: str, start: dt.datetime, end: dt.datetime) -> float:
    events = db.fetch_events(store_id, start, end, include_staff=False)
    uv = len({e["visitor_id"] for e in events if e["event_type"] == "ENTRY"})
    if not uv:
        return 0.0
    conv = pos.correlate_conversions(store_id, start, end)
    return len(conv["converted_visitors"]) / uv


def _dead_zone(
    store_id: str,
    events: list[dict[str, Any]],
    start: Optional[dt.datetime],
    end: Optional[dt.datetime],
) -> list[dict[str, Any]]:
    settings = get_settings()
    if start is None or end is None:
        return []
    window = dt.timedelta(minutes=settings.dead_zone_window_min)

    activity = sorted(e["timestamp"] for e in events if e["event_type"] in ACTIVITY_TYPES)
    marks = [start] + activity + [end]
    out = []
    for a, b in zip(marks, marks[1:]):
        gap = b - a
        if gap < window:
            continue
        midpoint = a + gap / 2
        if not layout.is_open_at(store_id, midpoint):
            continue
        out.append(
            {
                "type": "DEAD_ZONE",
                "severity": "WARN",
                "detected_at": _iso(b),
                "detail": f"No customer activity for {int(gap.total_seconds() // 60)} min "
                f"(>= {settings.dead_zone_window_min} min) during open hours.",
                "suggested_action": "Verify the camera feed is live and check for floor obstructions.",
            }
        )
    return out[:1]  # one representative dead-zone per window keeps the response actionable


def _iso(d: dt.datetime) -> str:
    return d.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
