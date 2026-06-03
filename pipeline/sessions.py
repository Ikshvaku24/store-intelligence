"""Visitor-id assignment, re-entry, staff cascade, cross-camera linking
(BUILD_SPEC Sections 7.5 and 9.2).

Pure Python (embeddings are passed in as plain float lists) so the identity and
staff logic is unit-testable without torch/torchreid. The heavy OSNet/HSV
feature extraction lives in pipeline/reid.py and feeds this module.

visitor_id semantics (BUILD_SPEC Section 6.4): a stable physical-person token
within the clip window. A re-entry REUSES the token and emits REENTRY -- it does
not mint a new id and does not increment unique visitors.
"""
from __future__ import annotations

import datetime as dt
import math
import uuid
from dataclasses import dataclass, field
from typing import Optional, Sequence

Vector = Sequence[float]


def cosine_similarity(a: Vector, b: Vector) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class _GalleryEntry:
    visitor_id: str
    embedding: list[float]
    histogram: Optional[list[float]]
    exit_ts: dt.datetime


@dataclass
class Assignment:
    visitor_id: str
    id_source: str          # within_camera | reentry_match | cross_camera_match
    id_confidence: float
    is_reentry: bool = False
    suppress_entry: bool = False  # set when a cross-camera duplicate is detected


@dataclass
class SessionManager:
    reentry_ttl_s: int = 600                 # 10 min gallery TTL
    reentry_threshold: float = 0.60          # cosine match for re-entry
    cross_camera_threshold: float = 0.75     # stricter; still flagged low-confidence
    cross_camera_window_s: int = 5           # synchronized +/- window

    _track_to_visitor: dict[tuple[str, int], str] = field(default_factory=dict)
    _gallery: list[_GalleryEntry] = field(default_factory=list)
    _cross_active: list[_GalleryEntry] = field(default_factory=list)
    _staff: set[str] = field(default_factory=set)
    _seq: dict[str, int] = field(default_factory=dict)

    # ---- visitor assignment ------------------------------------------------
    def assign(
        self,
        camera_id: str,
        track_id: int,
        role: str,
        embedding: Optional[Vector] = None,
        histogram: Optional[Vector] = None,
        ts: Optional[dt.datetime] = None,
    ) -> Assignment:
        key = (camera_id, track_id)
        # 1. Already mapped within this camera.
        if key in self._track_to_visitor:
            return Assignment(self._track_to_visitor[key], "within_camera", 0.95)

        # 2. Re-entry (ENTRY camera only): match the recently-exited gallery.
        if role == "ENTRY" and embedding is not None and ts is not None:
            match = self._match_gallery(embedding, histogram, ts)
            if match is not None:
                vid, conf = match
                self._track_to_visitor[key] = vid
                return Assignment(vid, "reentry_match", conf, is_reentry=True)

        # 3. Cross-camera duplicate (best-effort, always flagged low-confidence).
        if embedding is not None and ts is not None:
            xc = self._match_cross_camera(embedding, ts)
            if xc is not None:
                vid, conf = xc
                self._track_to_visitor[key] = vid
                # Entry counting is single-camera, so a cross-camera dup never
                # mints an ENTRY; suppress it and flag the low confidence.
                return Assignment(vid, "cross_camera_match", conf, suppress_entry=True)

        # 4. Genuinely new.
        vid = f"VIS_{uuid.uuid4().hex[:8]}"
        self._track_to_visitor[key] = vid
        if embedding is not None and ts is not None:
            self._cross_active.append(_GalleryEntry(vid, list(embedding), _as_list(histogram), ts))
        return Assignment(vid, "within_camera", 0.9)

    def _match_gallery(
        self, embedding: Vector, histogram: Optional[Vector], ts: dt.datetime
    ) -> Optional[tuple[str, float]]:
        best: Optional[tuple[str, float]] = None
        for entry in self._gallery:
            if (ts - entry.exit_ts).total_seconds() > self.reentry_ttl_s:
                continue
            sim = cosine_similarity(embedding, entry.embedding)
            # Histogram agreement is required as a second signal to reject
            # similar-clothing collisions at the door (BUILD_SPEC Section 9.3).
            hist_ok = True
            if histogram is not None and entry.histogram is not None:
                hist_ok = cosine_similarity(histogram, entry.histogram) >= 0.5
            if sim >= self.reentry_threshold and hist_ok:
                if best is None or sim > best[1]:
                    best = (entry.visitor_id, round(sim, 3))
        return best

    def _match_cross_camera(
        self, embedding: Vector, ts: dt.datetime
    ) -> Optional[tuple[str, float]]:
        best: Optional[tuple[str, float]] = None
        for entry in self._cross_active:
            if abs((ts - entry.exit_ts).total_seconds()) > self.cross_camera_window_s:
                continue
            sim = cosine_similarity(embedding, entry.embedding)
            if sim >= self.cross_camera_threshold:
                if best is None or sim > best[1]:
                    # Deliberately low reported confidence: this signal is the
                    # least reliable one on this footage.
                    best = (entry.visitor_id, round(min(sim, 0.5), 3))
        return best

    def on_exit(self, visitor_id: str, embedding: Optional[Vector], histogram: Optional[Vector], ts: dt.datetime) -> None:
        if embedding is None:
            return
        self._gallery.append(_GalleryEntry(visitor_id, list(embedding), _as_list(histogram), ts))

    # ---- staff cascade -----------------------------------------------------
    def mark_staff(self, visitor_id: str) -> None:
        self._staff.add(visitor_id)

    def is_staff(self, visitor_id: str) -> bool:
        return visitor_id in self._staff

    def staff_ids(self) -> set[str]:
        """The position-flagged staff so far (backroom/behind-counter/vanity)."""
        return set(self._staff)

    # ---- session sequence --------------------------------------------------
    def next_seq(self, visitor_id: str) -> int:
        self._seq[visitor_id] = self._seq.get(visitor_id, 0) + 1
        return self._seq[visitor_id]


def staff_by_zone(zone_flags: dict[str, bool], role: str) -> bool:
    """Position-first staff rule used by the orchestrator (BUILD_SPEC Section 9.2):
    backroom presence, or a foot-point inside a staff zone, => staff."""
    if role == "BACKROOM":
        return True
    return bool(zone_flags.get("is_staff_zone"))


def _as_list(v: Optional[Vector]) -> Optional[list[float]]:
    return list(v) if v is not None else None
