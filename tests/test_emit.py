# PROMPT: "Write pytest tests for EventWriter.finalize_staff (re-stamps is_staff
#   from the resolved staff set across all of a visitor's events) and
#   drop_short_tracks (removes flicker-track events but keeps ENTRY/EXIT/REENTRY,
#   which the tripwire debounce already guards)."
# CHANGES MADE: Added the assertion that finalize_staff also UNSETS is_staff for
#   visitors not in the set (the 'everyone-not-staff-is-customer' rule), and a case
#   proving a short ENTRY track is kept while a short ZONE_ENTER track is dropped.
from __future__ import annotations

import datetime as dt

from pipeline.emit import EventWriter, build_event

T0 = dt.datetime(2026, 4, 10, 14, 40, 0, tzinfo=dt.timezone.utc)
KEEP = {"ENTRY", "EXIT", "REENTRY"}


def _ev(vid, etype="ZONE_ENTER", is_staff=False):
    zone = None if etype in KEEP else "MAKEUP_LIPS"
    return build_event("S", "CAM_MAKEUP_02", vid, etype, T0, 0.8, zone_id=zone, is_staff=is_staff)


def test_finalize_staff_sets_and_unsets():
    w = EventWriter()
    w.add(_ev("a"))
    w.add(_ev("a", etype="ZONE_EXIT"))
    w.add(_ev("b", is_staff=True))   # wrongly staff at emit time
    flipped = w.finalize_staff({"a"})
    state = {(e.visitor_id, e.event_type.value): e.is_staff for e in w._events}
    assert state[("a", "ZONE_ENTER")] is True
    assert state[("a", "ZONE_EXIT")] is True
    assert state[("b", "ZONE_ENTER")] is False   # not in staff set -> customer
    assert flipped == 3


def test_drop_short_tracks_keeps_entry_drops_zone():
    w = EventWriter()
    w.add(_ev("ghostzone", "ZONE_ENTER"))
    w.add(_ev("ghostentry", "ENTRY"))
    removed = w.drop_short_tracks({"ghostzone", "ghostentry"}, KEEP)
    types = {e.event_type.value for e in w._events}
    assert removed == 1
    assert "ENTRY" in types and "ZONE_ENTER" not in types


def test_drop_short_tracks_noop_when_empty():
    w = EventWriter()
    w.add(_ev("x"))
    assert w.drop_short_tracks(set(), KEEP) == 0
    assert len(w._events) == 1
