"""Per-zone heatmap (BUILD_SPEC Section 10.4).

Returns per-zone visit_count + avg_dwell_ms + a 0-100 score normalised across
zones for grid rendering. Flags data_confidence=LOW below the configured
session threshold so a thin window isn't over-interpreted.
"""
from __future__ import annotations

from statistics import mean
from typing import Any, Optional

from app import db
from app.config import get_settings
from app.metrics import resolve_window


def compute_heatmap(store_id: str, window_min: Optional[int] = None) -> dict[str, Any]:
    settings = get_settings()
    start, end = resolve_window(store_id, window_min)
    events = db.fetch_events(store_id, start, end, include_staff=False)

    visits: dict[str, set[str]] = {}
    dwell: dict[str, list[int]] = {}
    for e in events:
        zone = e.get("zone_id")
        if not zone:
            continue
        if e["event_type"] == "ZONE_ENTER":
            visits.setdefault(zone, set()).add(e["visitor_id"])
        if e["event_type"] in {"ZONE_DWELL", "ZONE_EXIT"} and e.get("dwell_ms"):
            dwell.setdefault(zone, []).append(int(e["dwell_ms"]))

    zones = sorted(set(visits) | set(dwell))
    raw = {z: len(visits.get(z, set())) for z in zones}
    max_visits = max(raw.values()) if raw else 0

    zone_rows = []
    for z in zones:
        vc = raw[z]
        avg_dwell = int(mean(dwell[z])) if dwell.get(z) else 0
        score = round((vc / max_visits) * 100, 1) if max_visits else 0.0
        zone_rows.append(
            {"zone_id": z, "visit_count": vc, "avg_dwell_ms": avg_dwell, "score_0_100": score}
        )

    sessions = len({e["visitor_id"] for e in events if e["event_type"] == "ENTRY"})
    data_confidence = "OK" if sessions >= settings.heatmap_min_sessions else "LOW"

    return {
        "store_id": store_id,
        "zones": zone_rows,
        "session_count": sessions,
        "data_confidence": data_confidence,
        "window": {
            "start": start.isoformat().replace("+00:00", "Z") if start else None,
            "end": end.isoformat().replace("+00:00", "Z") if end else None,
            "window_min": window_min,
        },
    }
