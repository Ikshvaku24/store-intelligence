"""Schema-normalisation adapter (BUILD_SPEC Section 6, extended 2026-06-03).

The hackathon's authoritative event stream emits **three differently-shaped
events with inconsistent field names** -- this adapter maps each onto the single
canonical row the metrics/funnel/heatmap layer already understands, so the graded
API ingests the provided ``sample_events.jsonl`` (and the held-out events) without
the downstream code caring about the wire shape.

Wire shapes seen in the provided sample
---------------------------------------
* entry / exit            -> ``id_token``, ``store_code``, ``event_timestamp``,
                             ``gender_pred``/``age_pred``/``age_bucket``/
                             ``is_face_hidden``/``group_id``/``group_size``
* zone_entered / _exited  -> ``track_id``, ``store_id``, ``event_time``,
                             ``zone_name``/``zone_type``/``is_revenue_zone``/
                             ``zone_hotspot_x,y`` + demographics
* queue_completed/_abandoned -> ``track_id``, ``queue_join_ts``/``served_ts``/
                             ``exit_ts``, ``wait_seconds``/``queue_position_at_join``/
                             ``abandoned``

Mapping decisions
-----------------
* **event_type** is folded to the canonical upper-case vocabulary
  (``ENTRY``/``EXIT``/``ZONE_ENTER``/``ZONE_EXIT``/``BILLING_QUEUE_JOIN``/
  ``BILLING_QUEUE_ABANDON``) so existing aggregations keep working unchanged.
* **identity is not unified across cameras** in the source (entry uses a string
  ``id_token``; zone/queue use an integer ``track_id``). We honour that: entry
  tokens pass through; per-camera ``track_id``s are namespaced ``T{n}@{camera}``
  so the same integer on two cameras is not collapsed into one visitor.
* **store_id is canonicalised** (``store_1076`` and ``ST1076`` are the same store
  in the sample) so entry events (``store_code``) and zone events (``store_id``)
  land under one id. Applied symmetrically on read (see ``app/db``).
* **event_id is synthesised deterministically** when the source has none, so
  re-POSTing the same event is still idempotent (conflict-ignore on event_id).
* the rich extra fields (demographics, group, zone_type/is_revenue, queue
  timings) are carried into ``metadata`` for the new analytics, never dropped.

Already-canonical events (our own pipeline, the existing test-suite) are detected
by their event_type and **passed through untouched**.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

# new wire event_type -> canonical internal event_type
_EVENT_TYPE_MAP = {
    "entry": "ENTRY",
    "exit": "EXIT",
    "zone_entered": "ZONE_ENTER",
    "zone_exited": "ZONE_EXIT",
    "queue_completed": "BILLING_QUEUE_JOIN",
    "queue_abandoned": "BILLING_QUEUE_ABANDON",
}
NEW_EVENT_TYPES = frozenset(_EVENT_TYPE_MAP)

_STORE_RE = re.compile(r"^store[_-]?(\d+)$", re.IGNORECASE)


def canonical_store_id(store: Any) -> Any:
    """Fold equivalent store identifiers to one canonical form.

    ``store_1076`` / ``STORE_1076`` -> ``ST1076``; an ``ST####`` or any other id
    is returned stripped/unchanged (so ``STORE_BLR_002`` is left intact).
    """
    if not isinstance(store, str):
        return store
    s = store.strip()
    m = _STORE_RE.match(s)
    if m:
        return "ST" + m.group(1)
    return s


def _det_event_id(*parts: Any) -> str:
    """Deterministic event_id from identifying parts (stable -> idempotent)."""
    raw = "|".join("" if p is None else str(p) for p in parts)
    return "EVT_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def _yes(v: Any) -> Any:
    """Map a 'Yes'/'No' flag to bool; pass through anything else unchanged."""
    if isinstance(v, str):
        if v.strip().lower() in {"yes", "true", "y", "1"}:
            return True
        if v.strip().lower() in {"no", "false", "n", "0"}:
            return False
    return v


def _clean(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if v is not None}


def to_canonical(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Map one wire event to a list of canonical StoreEvent dicts.

    Returns ``[raw]`` unchanged for already-canonical / unrecognised events so the
    strict ``StoreEvent`` validator decides their fate (preserving partial-success
    behaviour for genuinely malformed events).
    """
    et = raw.get("event_type")
    if et not in NEW_EVENT_TYPES:
        return [raw]  # canonical pipeline event or malformed -> strict validation

    canonical_type = _EVENT_TYPE_MAP[et]
    store_id = canonical_store_id(raw.get("store_id") or raw.get("store_code"))
    camera_id = raw.get("camera_id")

    # demographics are shared across shapes under two naming conventions.
    demo = {
        "gender": raw.get("gender_pred") if raw.get("gender_pred") is not None else raw.get("gender"),
        "age": raw.get("age_pred") if raw.get("age_pred") is not None else raw.get("age"),
        "age_bucket": raw.get("age_bucket"),
        "is_face_hidden": raw.get("is_face_hidden"),
        "source_event_type": et,
    }

    if et in {"entry", "exit"}:
        visitor_id = raw.get("id_token")
        ts = raw.get("event_timestamp")
        meta = _clean({
            **demo,
            "id_source": "within_camera",
            "group_id": raw.get("group_id"),
            "group_size": raw.get("group_size"),
        })
        event_id = _det_event_id(store_id, camera_id, visitor_id, canonical_type, ts)
        return [_row(event_id, store_id, camera_id, visitor_id, canonical_type, ts,
                     None, bool(raw.get("is_staff", False)), meta)]

    if et in {"zone_entered", "zone_exited"}:
        track = raw.get("track_id")
        visitor_id = f"T{track}@{camera_id}" if track is not None else None
        ts = raw.get("event_time")
        meta = _clean({
            **demo,
            "track_id": track,
            "zone_name": raw.get("zone_name"),
            "zone_type": raw.get("zone_type"),
            "is_revenue_zone": _yes(raw.get("is_revenue_zone")),
            "zone_hotspot_x": raw.get("zone_hotspot_x"),
            "zone_hotspot_y": raw.get("zone_hotspot_y"),
        })
        event_id = _det_event_id(store_id, camera_id, visitor_id, canonical_type, ts, raw.get("zone_id"))
        return [_row(event_id, store_id, camera_id, visitor_id, canonical_type, ts,
                     raw.get("zone_id"), False, meta)]

    # queue_completed / queue_abandoned
    track = raw.get("track_id")
    visitor_id = f"T{track}@{camera_id}" if track is not None else None
    ts = raw.get("queue_join_ts")  # window membership is anchored on the join moment
    meta = _clean({
        **demo,
        "track_id": track,
        "zone_name": raw.get("zone_name"),
        "zone_type": raw.get("zone_type"),
        "is_revenue_zone": _yes(raw.get("is_revenue_zone")),
        "queue_depth": raw.get("queue_position_at_join"),
        "queue_position_at_join": raw.get("queue_position_at_join"),
        "wait_seconds": raw.get("wait_seconds"),
        "abandoned": raw.get("abandoned"),
        "queue_served_ts": raw.get("queue_served_ts"),
        "queue_exit_ts": raw.get("queue_exit_ts"),
    })
    event_id = raw.get("queue_event_id") or _det_event_id(
        store_id, camera_id, visitor_id, canonical_type, ts
    )
    return [_row(event_id, store_id, camera_id, visitor_id, canonical_type, ts,
                 raw.get("zone_id"), False, meta)]


def _row(event_id, store_id, camera_id, visitor_id, event_type, timestamp,
         zone_id, is_staff, metadata) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": timestamp,
        "zone_id": zone_id,
        "dwell_ms": 0,
        "is_staff": is_staff,
        "confidence": 1.0,
        "metadata": metadata,
    }
