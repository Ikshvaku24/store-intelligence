"""Batch event ingestion with per-event validation, idempotent storage, and
partial success (BUILD_SPEC Section 10.1).

Contract:
* up to ``max_batch_size`` events per call (else a structured rejection);
* each event validated independently -- one malformed event never sinks the batch;
* storage is idempotent: conflict-ignore on ``event_id`` means re-POSTing the same
  payload yields ``accepted: 0, duplicates: N`` and identical DB state.
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app import db
from app.config import get_settings
from app.models import StoreEvent
from app.normalize import to_canonical


def ingest_batch(raw_events: list[dict[str, Any]], trace_id: str) -> dict[str, Any]:
    settings = get_settings()
    received = len(raw_events)

    if received > settings.max_batch_size:
        return {
            "error": "batch_too_large",
            "detail": f"Batch of {received} exceeds limit of {settings.max_batch_size}.",
            "received": received,
            "accepted": 0,
            "duplicates": 0,
            "rejected": received,
            "trace_id": trace_id,
            "_status": 413,
        }

    valid: list[StoreEvent] = []
    errors: list[dict[str, Any]] = []

    for idx, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            errors.append({"index": idx, "event_id": None, "reason": "event must be a JSON object"})
            continue
        # Normalise the wire shape (entry/zone/queue) into canonical rows; an
        # already-canonical event is returned unchanged. One source event maps to
        # one canonical row, so a normalisation failure rejects just that event.
        try:
            canonical_rows = to_canonical(raw)
        except Exception as exc:  # noqa: BLE001 - defensive: never sink the batch
            errors.append({"index": idx, "event_id": _src_id(raw), "reason": f"normalise: {exc}"})
            continue

        sub_events: list[StoreEvent] = []
        reason: str | None = None
        for row in canonical_rows:
            try:
                sub_events.append(StoreEvent.model_validate(row))
            except ValidationError as exc:
                reason = _summarise(exc)
                break
        if reason is not None:
            errors.append({"index": idx, "event_id": _src_id(raw), "reason": reason})
        else:
            valid.extend(sub_events)

    # De-duplicate within the batch (keep first), then split DB-existing.
    seen: set[str] = set()
    batch_dupes = 0
    deduped: list[StoreEvent] = []
    for ev in valid:
        if ev.event_id in seen:
            batch_dupes += 1
            continue
        seen.add(ev.event_id)
        deduped.append(ev)

    already = db.existing_event_ids(seen)
    to_insert = [ev for ev in deduped if ev.event_id not in already]
    db.insert_events_ignore_conflicts([ev.to_row() for ev in to_insert])

    duplicates = batch_dupes + len(already)
    accepted = len(to_insert)
    rejected = len(errors)

    return {
        "received": received,
        "accepted": accepted,
        "duplicates": duplicates,
        "rejected": rejected,
        "errors": errors,
        "trace_id": trace_id,
        "_status": 200,
    }


def _src_id(raw: dict[str, Any]) -> Any:
    """Best-effort source identifier for an error row (canonical or new shape)."""
    return raw.get("event_id") or raw.get("queue_event_id") or raw.get("id_token")


def _summarise(exc: ValidationError) -> str:
    parts = []
    for e in exc.errors()[:5]:
        loc = ".".join(str(p) for p in e.get("loc", ()))
        parts.append(f"{loc}: {e.get('msg')}")
    return "; ".join(parts) or "validation error"
