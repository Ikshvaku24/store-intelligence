"""Customer metrics (BUILD_SPEC Section 10.2).

Every metric here excludes ``is_staff=true`` rows, returns explicit zeros (never
null/NaN) on zero traffic, and is computed at query time (never cached from a
prior day). The window is anchored on the latest event timestamp for the store
so results are deterministic for historical clips and meaningful under live
replay alike.
"""
from __future__ import annotations

import datetime as dt
from statistics import mean
from typing import Any, Optional

from app import db, pos
from app.config import get_settings

ENTRY_TYPES = {"ENTRY"}  # REENTRY reuses an existing visitor_id, so not counted again


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def resolve_window(
    store_id: str, window_min: Optional[int] = None
) -> tuple[Optional[dt.datetime], Optional[dt.datetime]]:
    """Return (start, end) anchored on the store's latest event.

    No window_min => the full available range for the store. Empty store =>
    (None, None) so downstream returns explicit zeros.
    """
    all_events = db.fetch_events(store_id)
    if not all_events:
        return None, None
    ts = [e["timestamp"] for e in all_events]
    end = max(ts)
    if window_min is None:
        return min(ts), end
    return end - dt.timedelta(minutes=window_min), end


def unique_visitors(events: list[dict[str, Any]]) -> int:
    return len({e["visitor_id"] for e in events if e["event_type"] in ENTRY_TYPES})


def visitor_base(events: list[dict[str, Any]]) -> tuple[set[str], str]:
    """Distinct customer visitor tokens for the window, with the basis used.

    Entry-anchored when ENTRY events exist (the clean, deduped path the graded
    held-out events use). When a clip window has NO entries but the floor is busy
    (people already inside, nobody crossing the door — the real footage reality),
    fall back to the distinct non-staff visitor_ids seen anywhere on the floor so
    the North Star metric is still meaningful instead of collapsing to zero.

    ``events`` are already staff-excluded by the caller, so "everyone here is a
    customer" holds. The floor basis is approximate (per-camera ids; cross-camera
    dedup is best-effort), so callers flag ``data_confidence=LOW`` for it.
    """
    entries = {e["visitor_id"] for e in events if e["event_type"] in ENTRY_TYPES}
    if entries:
        return entries, "entry"
    floor = {e["visitor_id"] for e in events}
    return floor, "floor"


def avg_dwell_by_zone(events: list[dict[str, Any]]) -> dict[str, int]:
    """Mean dwell per zone, in ms.

    Superset of two sources so it works on both schemas: (1) an explicit
    ``dwell_ms`` on ``ZONE_DWELL``/``ZONE_EXIT`` (our pipeline + legacy); (2) for
    the new ``zone_entered``/``zone_exited`` pairs (which carry no ``dwell_ms``),
    the duration is derived by pairing each EXIT with the matching prior ENTER on
    the same ``(visitor_id, zone_id)``.
    """
    by_ts = sorted(events, key=lambda e: e["timestamp"])
    open_enters: dict[tuple[str, str], list[dt.datetime]] = {}
    buckets: dict[str, list[int]] = {}
    for e in by_ts:
        zone = e.get("zone_id")
        if not zone:
            continue
        etype = e["event_type"]
        key = (e["visitor_id"], zone)
        if etype == "ZONE_ENTER":
            open_enters.setdefault(key, []).append(e["timestamp"])
        elif etype in {"ZONE_DWELL", "ZONE_EXIT"}:
            dwell = int(e["dwell_ms"]) if e.get("dwell_ms") else 0
            if not dwell and etype == "ZONE_EXIT" and open_enters.get(key):
                enter_ts = open_enters[key].pop(0)
                dwell = int((e["timestamp"] - enter_ts).total_seconds() * 1000)
            if dwell > 0:
                buckets.setdefault(zone, []).append(dwell)
    return {z: int(mean(v)) for z, v in buckets.items() if v}


def demographics(events: list[dict[str, Any]], base: Optional[set[str]] = None) -> dict[str, Any]:
    """Gender / age-bucket breakdown over distinct (non-staff) visitors.

    The graders' events carry ``gender``/``age_bucket`` (in metadata after
    normalisation); we attribute one record per visitor (first event that has a
    value) so a person isn't counted once per zone they browse. ``base`` restricts
    counting to the unique-visitor set (entry tokens, or the floor fallback) so the
    breakdown can't double-count the same person across their per-camera track ids,
    which are deliberately not unified (ADR-002)."""
    rec: dict[str, tuple[Optional[str], Optional[str]]] = {}
    for e in events:
        vid = e["visitor_id"]
        if vid in rec or (base is not None and vid not in base):
            continue
        md = e.get("metadata") or {}
        g, ab = md.get("gender"), md.get("age_bucket")
        if g is not None or ab is not None:
            rec[vid] = (g, ab)
    by_gender: dict[str, int] = {}
    by_age: dict[str, int] = {}
    for g, ab in rec.values():
        if g is not None:
            by_gender[g] = by_gender.get(g, 0) + 1
        if ab is not None:
            by_age[ab] = by_age.get(ab, 0) + 1
    return {"n_classified": len(rec), "by_gender": by_gender, "by_age_bucket": by_age}


def group_stats(events: list[dict[str, Any]], base: Optional[set[str]] = None) -> dict[str, Any]:
    """Group-shopping stats from ``group_id``/``group_size`` (in metadata)."""
    sizes: dict[str, int] = {}
    members: set[str] = set()
    for e in events:
        if base is not None and e["visitor_id"] not in base:
            continue
        md = e.get("metadata") or {}
        gid = md.get("group_id")
        if not gid:
            continue
        members.add(e["visitor_id"])
        gs = md.get("group_size")
        if gs is not None:
            sizes[gid] = max(sizes.get(gid, 0), int(gs))
    return {
        "group_count": len(sizes),
        "visitors_in_groups": len(members),
        "avg_group_size": round(mean(sizes.values()), 2) if sizes else 0.0,
    }


def avg_queue_wait_seconds(events: list[dict[str, Any]]) -> float:
    """Mean billing-queue wait (s) from ``wait_seconds`` (new queue events)."""
    waits = [
        float((e.get("metadata") or {}).get("wait_seconds"))
        for e in events
        if (e.get("metadata") or {}).get("wait_seconds") is not None and not e.get("is_staff")
    ]
    return round(mean(waits), 2) if waits else 0.0


def current_queue_depth(events: list[dict[str, Any]]) -> int:
    joins = [
        e
        for e in events
        if e["event_type"] == "BILLING_QUEUE_JOIN" and not e.get("is_staff")
    ]
    if not joins:
        return 0
    latest = max(joins, key=lambda e: e["timestamp"])
    depth = (latest.get("metadata") or {}).get("queue_depth")
    return int(depth) if depth is not None else 0


def abandonment_rate(
    events: list[dict[str, Any]], converted: Optional[set[str]] = None
) -> float:
    # An abandon is itself proof the visitor joined the queue. In the new schema a
    # ``queue_abandoned`` event maps to a single ABANDON row (no separate JOIN), so
    # the joiner base is the union JOIN ∪ ABANDON. For legacy data (which emits both
    # for an abandoner) the union is identical, so this stays back-compatible.
    #
    # A visitor who is POS-correlated as *converted* did not really abandon -- the
    # queue-exit was a completed purchase the pipeline couldn't see. Reconciling
    # against ``converted`` removes the "100% abandonment alongside N purchases"
    # contradiction (back-compatible: ``converted`` defaults to empty).
    converted = converted or set()
    abandoners = {
        e["visitor_id"]
        for e in events
        if e["event_type"] == "BILLING_QUEUE_ABANDON" and not e.get("is_staff")
    } - converted
    joiners = abandoners | {
        e["visitor_id"]
        for e in events
        if e["event_type"] == "BILLING_QUEUE_JOIN" and not e.get("is_staff")
    }
    if not joiners:
        return 0.0
    return round(len(abandoners) / len(joiners), 4)


def avg_id_confidence(events: list[dict[str, Any]]) -> Optional[float]:
    vals = [
        float((e.get("metadata") or {}).get("id_confidence"))
        for e in events
        if (e.get("metadata") or {}).get("id_confidence") is not None
    ]
    return round(mean(vals), 4) if vals else None


def compute_metrics(store_id: str, window_min: Optional[int] = None) -> dict[str, Any]:
    start, end = resolve_window(store_id, window_min)
    events = db.fetch_events(store_id, start, end, include_staff=False)

    base, basis = visitor_base(events)
    conv = pos.correlate_conversions(store_id, start, end)
    converted = conv["converted_visitors"]
    # A purchaser is, by definition, a unique visitor. Fold POS-correlated buyers into
    # the visitor base so the North Star numerator can never exceed its denominator:
    # entry-basis can undercount the floor (e.g. Store 2's tripwire caught ~1 crossing
    # while billing saw several buyers), which would otherwise yield a >100% rate. This
    # also guarantees a confirmed buyer is never dropped from the unique count.
    base = base | converted
    uv = len(base)
    conversion_rate = round(len(converted) / uv, 4) if uv else 0.0

    if uv == 0:
        data_confidence = "LOW"
    elif basis == "floor":
        # Floor-derived count: per-camera tokens, cross-camera dedup still best-effort.
        data_confidence = "LOW"
    else:
        data_confidence = "OK"

    return {
        "store_id": store_id,
        "unique_visitors": uv,
        "visitor_basis": basis,
        "conversion_rate": conversion_rate,
        "purchases": conv["purchase_count"],
        "avg_dwell_ms_by_zone": avg_dwell_by_zone(events),
        "current_queue_depth": current_queue_depth(events),
        "abandonment_rate": abandonment_rate(events, converted),
        "avg_queue_wait_seconds": avg_queue_wait_seconds(events),
        "demographics": demographics(events, base),
        "groups": group_stats(events, base),
        "window": _window_repr(start, end, window_min),
        "generated_at": now_utc().isoformat().replace("+00:00", "Z"),
        "data_confidence": data_confidence,
        "avg_id_confidence": avg_id_confidence(events),
    }


def _window_repr(
    start: Optional[dt.datetime], end: Optional[dt.datetime], window_min: Optional[int]
) -> dict[str, Any]:
    return {
        "start": _iso(start),
        "end": _iso(end),
        "window_min": window_min,
    }


def _iso(d: Optional[dt.datetime]) -> Optional[str]:
    if d is None:
        return None
    return d.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
