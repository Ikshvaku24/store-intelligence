"""Build + Pydantic-validate events, write events.jsonl (BUILD_SPEC Section 7.6).

Every event is validated against the shared StoreEvent schema (app/models.py)
*before* being written, so the pipeline fails loudly on a bad event rather than
the API silently rejecting it later.
"""
from __future__ import annotations

import datetime as dt
import json
import uuid
from typing import Any, Optional

from app.models import StoreEvent


def frame_timestamp(start_ts: dt.datetime, frame_index: int, fps: float) -> dt.datetime:
    """Deterministic event timestamp = start_ts + frame_index / fps (UTC)."""
    return start_ts + dt.timedelta(seconds=frame_index / fps)


def build_event(
    store_id: str,
    camera_id: str,
    visitor_id: str,
    event_type: str,
    timestamp: dt.datetime,
    confidence: float,
    zone_id: Optional[str] = None,
    dwell_ms: int = 0,
    is_staff: bool = False,
    metadata: Optional[dict[str, Any]] = None,
) -> StoreEvent:
    return StoreEvent.model_validate(
        {
            "event_id": str(uuid.uuid4()),
            "store_id": store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": event_type,
            "timestamp": timestamp,
            "zone_id": zone_id,
            "dwell_ms": dwell_ms,
            "is_staff": is_staff,
            "confidence": round(float(confidence), 4),
            "metadata": metadata or {},
        }
    )


class EventWriter:
    """Collects validated events and writes them sorted by timestamp."""

    def __init__(self) -> None:
        self._events: list[StoreEvent] = []

    def add(self, event: StoreEvent) -> None:
        self._events.append(event)

    def __len__(self) -> int:
        return len(self._events)

    def finalize_staff(self, staff_ids: set[str]) -> int:
        """Stamp is_staff on every event from the resolved staff set.

        Staff is decided per-visitor only after the whole run (position + VLM +
        heuristic), so we re-stamp here: a visitor in ``staff_ids`` is staff on all
        their events, everyone else is a customer. Returns the count flipped."""
        flipped = 0
        for ev in self._events:
            should = ev.visitor_id in staff_ids
            if ev.is_staff != should:
                ev.is_staff = should
                flipped += 1
        return flipped

    def drop_short_tracks(self, short_ids: set[str], keep_types: set[str]) -> int:
        """Remove events from flicker tracks (a visitor observed for too few
        frames), except ``keep_types``. ENTRY/EXIT/REENTRY are kept because the
        tripwire debounce already guards them. Returns the number removed."""
        if not short_ids:
            return 0
        before = len(self._events)
        self._events = [
            e
            for e in self._events
            if e.visitor_id not in short_ids or e.event_type.value in keep_types
        ]
        return before - len(self._events)

    def relabel_visitors(self, mapping: dict[str, str]) -> int:
        """Rewrite visitor_id via a dedup map (token -> canonical). Returns count
        changed. Used to collapse cross-camera duplicates and track fragments."""
        if not mapping:
            return 0
        changed = 0
        for ev in self._events:
            new = mapping.get(ev.visitor_id)
            if new is not None and new != ev.visitor_id:
                ev.visitor_id = new
                changed += 1
        return changed

    def write(self, path: str) -> int:
        self._events.sort(key=lambda e: e.timestamp)
        with open(path, "w", encoding="utf-8") as fh:
            for ev in self._events:
                row = ev.model_dump(mode="json")
                # Emit ISO-8601 Z form for the timestamp.
                row["timestamp"] = ev.timestamp.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
                fh.write(json.dumps(row) + "\n")
        return len(self._events)
