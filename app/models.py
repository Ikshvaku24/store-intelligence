"""Pydantic event schema -- the shared contract between the detection pipeline
and the graded API (BUILD_SPEC Section 6).

The required top-level keys and types match the challenge schema exactly. Two
*optional* metadata keys (`id_source`, `id_confidence`) are added so the API can
surface -- never hide -- unreliable cross-camera identity. Extra optional keys
are schema-permitted; the required keys remain present and correctly typed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    REENTRY = "REENTRY"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"


class IdSource(str, Enum):
    WITHIN_CAMERA = "within_camera"
    REENTRY_MATCH = "reentry_match"
    CROSS_CAMERA_MATCH = "cross_camera_match"


class EventMetadata(BaseModel):
    """Optional, additive metadata. All keys default to None so an event with an
    empty metadata object is valid."""

    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = None
    # --- identity transparency (additive, non-breaking) ---
    id_source: Optional[IdSource] = None
    id_confidence: Optional[float] = None

    @field_validator("id_confidence")
    @classmethod
    def _conf_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("id_confidence must be within [0, 1]")
        return v

    model_config = {"extra": "allow"}  # tolerate forward-compatible metadata keys


class StoreEvent(BaseModel):
    event_id: str = Field(..., min_length=1)
    store_id: str = Field(..., min_length=1)
    camera_id: str = Field(..., min_length=1)
    visitor_id: str = Field(..., min_length=1)
    event_type: EventType
    timestamp: datetime
    zone_id: Optional[str] = None
    dwell_ms: int = 0
    is_staff: bool = False
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("timestamp")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        """Normalise to timezone-aware UTC. Naive timestamps are assumed UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @field_validator("dwell_ms")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("dwell_ms must be non-negative")
        return v

    def to_row(self) -> dict[str, Any]:
        """Flatten to a DB row. Timestamp stored as naive-UTC for cross-dialect
        comparability (see app/db.py)."""
        ts = self.timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        return {
            "event_id": self.event_id,
            "store_id": self.store_id,
            "camera_id": self.camera_id,
            "visitor_id": self.visitor_id,
            "event_type": self.event_type.value,
            "timestamp": ts,
            "zone_id": self.zone_id,
            "dwell_ms": self.dwell_ms,
            "is_staff": self.is_staff,
            "confidence": self.confidence,
            "metadata": self.metadata.model_dump(exclude_none=True, mode="json"),
        }
