# PROMPT: "Write pytest tests for the pipeline staff resolver: a position-flagged
#   visitor is always staff; a person who is present across the clip and stands
#   close to several different people (serving) is flagged staff by the heuristic
#   while a brief lone shopper is not; and an injected VLM verdict overrides the
#   heuristic in both directions. Use a fake VLM so no network/Gemini is needed."
# CHANGES MADE: The AI tested only the heuristic. I added the VLM-precedence cases
#   (staff AND customer verdicts), the 'everyone-not-staff-is-customer' default,
#   and a presence_fraction unit check. The fake VLM mirrors the real classify()
#   signature so the resolver contract is exercised exactly as in production.
from __future__ import annotations

import datetime as dt

from pipeline.staff import StaffEvidence, resolve_staff

T0 = dt.datetime(2026, 4, 10, 14, 40, 0, tzinfo=dt.timezone.utc)


def _obs(ev, vid, sec, camera="CAM_MAKEUP_02", zones=("MAKEUP_LIPS",), near=(), crop=None):
    ev.observe(vid, T0 + dt.timedelta(seconds=sec), camera, zones, near,
               100.0 if crop else None, crop)


class _FakeVLM:
    def __init__(self, is_staff, conf=0.9):
        self.available = True
        self._v = {"is_staff": is_staff, "confidence": conf, "reason": "test"}

    def classify(self, visitor_id, crops_jpeg, context):
        return self._v


def test_position_staff_always_staff():
    ev = StaffEvidence()
    _obs(ev, "S1", 0, camera="CAM_BACK_04", zones=("BACKROOM",))
    staff, _ = resolve_staff(ev, position_staff={"S1"})
    assert "S1" in staff


def test_serving_many_persistent_is_staff():
    ev = StaffEvidence()
    for s in range(0, 100, 5):                      # present across the whole clip
        _obs(ev, "STAFF", s, near=("c1", "c2", "c3"))
    _obs(ev, "c1", 40)                              # a brief lone shopper
    staff, _ = resolve_staff(ev, position_staff=set())
    assert "STAFF" in staff
    assert "c1" not in staff                        # default: not staff => customer


def test_lone_shopper_is_customer():
    ev = StaffEvidence()
    _obs(ev, "c1", 0)
    _obs(ev, "c1", 2)
    staff, _ = resolve_staff(ev, position_staff=set())
    assert "c1" not in staff


def test_vlm_verdict_promotes_to_staff():
    ev = StaffEvidence()
    for s in (0, 10, 20):
        _obs(ev, "X", s, crop=b"jpeg")
    staff, log = resolve_staff(ev, position_staff=set(), vlm=_FakeVLM(is_staff=True))
    assert "X" in staff
    assert any(d["source"] == "vlm" and d["decision"] == "staff" for d in log)


def test_vlm_customer_overrides_heuristic():
    ev = StaffEvidence()
    # zones>=4 would make the heuristic say staff, but the VLM says customer -> wins
    for s in (0, 10, 20):
        _obs(ev, "Y", s, zones=("MAKEUP_LIPS", "MAKEUP_FOUNDATION", "FRAGRANCE_TABLE",
                                 "SKINCARE_CLEANSER"), crop=b"jpeg")
    staff, _ = resolve_staff(ev, position_staff=set(), vlm=_FakeVLM(is_staff=False, conf=0.8))
    assert "Y" not in staff


def test_presence_fraction():
    ev = StaffEvidence()
    _obs(ev, "A", 0)
    _obs(ev, "A", 100)        # spans the full observed camera window
    _obs(ev, "B", 0)          # only at the start
    assert ev.presence_fraction("A") == 1.0
    assert ev.presence_fraction("B") < 1.0
