"""Per-visitor staff resolution (BUILD_SPEC Section 9.2, extended).

The footage has **no uniform** and the backroom is empty, so position alone can't
catch staff working the floor (applying makeup, demonstrating products, operating
the POS). This module accumulates lightweight per-visitor evidence during the
pipeline run and resolves a final staff set with a layered cascade:

    1. position oracle   -- backroom / behind-counter / behind-vanity zones (hard)
    2. VLM confirmer     -- behavioural classification on ambiguous tracks (Gemini)
    3. heuristic fallback-- persistence + serve-many + zone/camera roaming

Resolution principle (per the user's instruction): **once a person is identified
as staff, everyone else is a customer.** So we only ever *add* to the staff set;
non-staff is the default. The API then filters ``is_staff=true`` out of every
customer metric.

Pure-Python (no cv2/torch) so it is unit-testable in the light venv. Crops are
passed in as already-encoded JPEG bytes; the VLM object is injected (and may be
None, in which case only position + heuristic are used).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol


class StaffVLM(Protocol):
    """Minimal interface the resolver needs from a VLM classifier."""

    available: bool

    def classify(
        self, visitor_id: str, crops_jpeg: list[bytes], context: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        ...


@dataclass
class VisitorEvidence:
    first_ts: dt.datetime
    last_ts: dt.datetime
    cameras: set[str] = field(default_factory=set)
    zones: set[str] = field(default_factory=set)
    near: set[str] = field(default_factory=set)   # distinct other visitors stood close to
    frames: int = 0
    crops: list[tuple[float, bytes]] = field(default_factory=list)  # (bbox_area, jpeg)
    emb_sum: list[float] = field(default_factory=list)  # running sum of OSNet embeddings
    emb_n: int = 0


class StaffEvidence:
    """Accumulates behavioural evidence per visitor across all clips."""

    def __init__(self, max_crops: int = 5) -> None:
        self.max_crops = max_crops
        self.visitors: dict[str, VisitorEvidence] = {}
        self._cam_first: dict[str, dt.datetime] = {}
        self._cam_last: dict[str, dt.datetime] = {}

    def observe(
        self,
        visitor_id: str,
        ts: dt.datetime,
        camera_id: str,
        zones: Iterable[str],
        near_visitors: Iterable[str] = (),
        bbox_area: Optional[float] = None,
        crop_jpeg: Optional[bytes] = None,
        embedding: Optional[Iterable[float]] = None,
    ) -> None:
        # Track each camera's observed time span (for presence_fraction).
        if camera_id not in self._cam_first or ts < self._cam_first[camera_id]:
            self._cam_first[camera_id] = ts
        if camera_id not in self._cam_last or ts > self._cam_last[camera_id]:
            self._cam_last[camera_id] = ts

        v = self.visitors.get(visitor_id)
        if v is None:
            v = VisitorEvidence(first_ts=ts, last_ts=ts)
            self.visitors[visitor_id] = v
        v.first_ts = min(v.first_ts, ts)
        v.last_ts = max(v.last_ts, ts)
        v.cameras.add(camera_id)
        v.zones.update(z for z in zones if z)
        v.near.update(n for n in near_visitors if n and n != visitor_id)
        v.frames += 1
        if crop_jpeg is not None:
            v.crops.append((float(bbox_area or 0.0), crop_jpeg))
            if len(v.crops) > self.max_crops:
                # keep the largest (clearest) crops
                v.crops.sort(key=lambda c: c[0], reverse=True)
                del v.crops[self.max_crops:]
        if embedding is not None:
            emb = list(embedding)
            if not v.emb_sum:
                v.emb_sum = emb
                v.emb_n = 1
            elif len(emb) == len(v.emb_sum):
                v.emb_sum = [s + e for s, e in zip(v.emb_sum, emb)]
                v.emb_n += 1

    def presence_fraction(self, visitor_id: str) -> float:
        """How much of its camera's observed span this visitor was present for."""
        v = self.visitors.get(visitor_id)
        if v is None:
            return 0.0
        spans = []
        for cam in v.cameras:
            cam_span = (self._cam_last[cam] - self._cam_first[cam]).total_seconds()
            if cam_span > 0:
                spans.append(cam_span)
        if not spans:
            return 0.0
        vis_span = (v.last_ts - v.first_ts).total_seconds()
        return min(1.0, vis_span / max(spans))

    def mean_embeddings(self) -> dict[str, list[float]]:
        """Mean OSNet embedding per visitor (only those with >=1 sample)."""
        return {
            vid: [s / v.emb_n for s in v.emb_sum]
            for vid, v in self.visitors.items()
            if v.emb_n > 0 and v.emb_sum
        }


# --- tunable heuristic thresholds (documented; conservative to avoid false staff) ---
HEUR_ZONES = 4            # touched >= this many distinct zones -> roams like staff
HEUR_NEAR = 3             # stood close to >= this many distinct people -> serving
HEUR_NEAR_PRESENCE = 0.6  # ...while present for >= this fraction of the clip
HEUR_MULTICAM_PRESENCE = 0.5  # appears on multiple cameras and persists


def _heuristic_is_staff(ev: StaffEvidence, vid: str) -> bool:
    v = ev.visitors[vid]
    pf = ev.presence_fraction(vid)
    if len(v.zones) >= HEUR_ZONES:
        return True
    if len(v.near) >= HEUR_NEAR and pf >= HEUR_NEAR_PRESENCE:
        return True
    if len(v.cameras) >= 2 and pf >= HEUR_MULTICAM_PRESENCE:
        return True
    return False


def resolve_staff(
    evidence: StaffEvidence,
    position_staff: Iterable[str],
    vlm: Optional[StaffVLM] = None,
    *,
    min_crops_for_vlm: int = 2,
    vlm_conf_threshold: float = 0.55,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Return (staff_ids, decision_log).

    Cascade per visitor: position (hard) -> VLM verdict (if available + enough
    crops) -> heuristic fallback. Everyone not added is a customer.
    """
    staff: set[str] = set(position_staff)
    log: list[dict[str, Any]] = []

    for vid, v in evidence.visitors.items():
        if vid in staff:
            log.append({"visitor_id": vid, "decision": "staff", "source": "position"})
            continue

        pf = round(evidence.presence_fraction(vid), 3)
        ctx = {
            "zones": sorted(v.zones),
            "cameras": sorted(v.cameras),
            "presence_fraction": pf,
            "near_count": len(v.near),
        }

        verdict = None
        if vlm is not None and getattr(vlm, "available", False) and len(v.crops) >= min_crops_for_vlm:
            crops = [c for _, c in sorted(v.crops, key=lambda c: c[0], reverse=True)]
            verdict = vlm.classify(vid, crops, ctx)

        if verdict is not None:
            conf = float(verdict.get("confidence", 0.0) or 0.0)
            if verdict.get("is_staff") and conf >= vlm_conf_threshold:
                staff.add(vid)
                log.append({"visitor_id": vid, "decision": "staff", "source": "vlm",
                            "confidence": conf, "reason": verdict.get("reason")})
            else:
                log.append({"visitor_id": vid, "decision": "customer", "source": "vlm",
                            "confidence": conf, "reason": verdict.get("reason")})
            continue

        if _heuristic_is_staff(evidence, vid):
            staff.add(vid)
            log.append({"visitor_id": vid, "decision": "staff", "source": "heuristic", **ctx})
        else:
            log.append({"visitor_id": vid, "decision": "customer", "source": "heuristic", **ctx})

    return staff, log
