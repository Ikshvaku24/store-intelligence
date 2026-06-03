# PROMPT: "Write pytest tests for the detection pipeline's pure-Python core:
#   a debounced tripwire that emits ENTRY on an inbound crossing and EXIT on an
#   outbound one; a session manager where a visitor who exits and re-enters gets
#   the SAME visitor_id (REENTRY) but two people with different appearance from the
#   same spot get DIFFERENT visitor_ids; and point-in-polygon zone assignment."
# CHANGES MADE: The AI's tripwire test asserted a crossing on the very first frame
#   on the inside, missing the K-frame debounce; I rewrote it to require min_frames
#   on each side and to test BOTH directions. I added the "two distinct embeddings
#   from the same spot => two visitor_ids" case (the brief's hardest re-entry edge),
#   which the AI omitted, and an emit.build_event schema-validation check so the
#   pipeline-to-API contract is exercised end to end.
from __future__ import annotations

import datetime as dt

from pipeline import geometry
from pipeline.emit import build_event, frame_timestamp
from pipeline.sessions import SessionManager, cosine_similarity, staff_by_zone

T0 = dt.datetime(2026, 4, 16, 8, 0, 0, tzinfo=dt.timezone.utc)


# --- geometry --------------------------------------------------------------
def test_point_in_polygon():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert geometry.point_in_polygon(square, [5, 5]) is True
    assert geometry.point_in_polygon(square, [15, 5]) is False
    assert geometry.point_in_polygon(square, [-1, -1]) is False


def test_zones_for_point():
    zones = {
        "A": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "B": [[20, 20], [30, 20], [30, 30], [20, 30]],
    }
    assert set(geometry.zones_for_point(zones, [5, 5])) == {"A"}
    assert set(geometry.zones_for_point(zones, [25, 25])) == {"B"}
    assert set(geometry.zones_for_point(zones, [100, 100])) == set()


def test_tripwire_both_directions():
    # Horizontal line y=0; inside is the y>0 (wood) side.
    inside_sign = geometry.inside_sign_from_point([0, 0], [10, 0], [5, 5])
    tw = geometry.TripwireCounter([0, 0], [10, 0], inside_sign, min_frames=2)

    # Establish outside (y<0) for 2 frames, then cross inbound (y>0) for 2 frames.
    assert tw.update(1, [5, -5]) is None
    assert tw.update(1, [5, -5]) is None
    assert tw.update(1, [5, 5]) is None
    assert tw.update(1, [5, 5]) == "ENTRY"

    # Now cross back outbound -> EXIT.
    assert tw.update(1, [5, -5]) is None
    assert tw.update(1, [5, -5]) == "EXIT"


def test_tripwire_debounce_rejects_single_frame_flicker():
    inside_sign = geometry.inside_sign_from_point([0, 0], [10, 0], [5, 5])
    tw = geometry.TripwireCounter([0, 0], [10, 0], inside_sign, min_frames=3)
    # 3 frames outside to set a stable side.
    for _ in range(3):
        assert tw.update(7, [5, -5]) is None
    # A single inside frame (flicker) must NOT trigger a crossing.
    assert tw.update(7, [5, 5]) is None


# --- session / identity ----------------------------------------------------
def test_within_camera_reuse():
    mgr = SessionManager()
    a1 = mgr.assign("CAM_SKIN_01", 1, "FLOOR_SKINCARE")
    a2 = mgr.assign("CAM_SKIN_01", 1, "FLOOR_SKINCARE")
    assert a1.visitor_id == a2.visitor_id
    assert a2.id_source == "within_camera"


def test_exit_then_reentry_same_visitor_id():
    mgr = SessionManager()
    emb = [1.0, 0.0, 0.0]
    first = mgr.assign("CAM_ENTRY_03", 1, "ENTRY", embedding=emb, ts=T0)
    mgr.on_exit(first.visitor_id, emb, None, T0 + dt.timedelta(seconds=10))

    # Same person returns 30s later (new track id) -> REENTRY, same visitor_id.
    again = mgr.assign("CAM_ENTRY_03", 2, "ENTRY", embedding=emb, ts=T0 + dt.timedelta(seconds=40))
    assert again.visitor_id == first.visitor_id
    assert again.is_reentry is True
    assert again.id_source == "reentry_match"


def test_two_distinct_people_same_spot_two_visitor_ids():
    mgr = SessionManager()
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.0, 1.0, 0.0]  # orthogonal -> cosine 0, no match
    a = mgr.assign("CAM_ENTRY_03", 1, "ENTRY", embedding=emb_a, ts=T0)
    mgr.on_exit(a.visitor_id, emb_a, None, T0 + dt.timedelta(seconds=3))
    b = mgr.assign("CAM_ENTRY_03", 2, "ENTRY", embedding=emb_b, ts=T0 + dt.timedelta(seconds=6))
    assert a.visitor_id != b.visitor_id
    assert b.is_reentry is False


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([], [1]) == 0.0


def test_staff_cascade_backroom_and_zone():
    assert staff_by_zone({}, "BACKROOM") is True
    assert staff_by_zone({"is_staff_zone": True}, "BILLING") is True
    assert staff_by_zone({"is_staff_zone": False}, "FLOOR_MAKEUP") is False

    mgr = SessionManager()
    a = mgr.assign("CAM_BACK_04", 1, "BACKROOM")
    mgr.mark_staff(a.visitor_id)
    assert mgr.is_staff(a.visitor_id) is True


def test_session_seq_increments():
    mgr = SessionManager()
    a = mgr.assign("CAM_ENTRY_03", 1, "ENTRY")
    assert mgr.next_seq(a.visitor_id) == 1
    assert mgr.next_seq(a.visitor_id) == 2


# --- emit / schema contract ------------------------------------------------
def test_build_event_validates_against_schema():
    ts = frame_timestamp(T0, frame_index=30, fps=15)  # +2.0s
    ev = build_event(
        store_id="STORE_BLR_002",
        camera_id="CAM_ENTRY_03",
        visitor_id="VIS_abc",
        event_type="ENTRY",
        timestamp=ts,
        confidence=0.81,
        metadata={"id_source": "within_camera", "id_confidence": 0.9, "session_seq": 1},
    )
    assert ev.event_type.value == "ENTRY"
    assert ev.timestamp == ts
    row = ev.to_row()
    assert row["store_id"] == "STORE_BLR_002"
    assert row["metadata"]["id_source"] == "within_camera"
