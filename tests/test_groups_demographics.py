"""Group detection + the per-visitor metadata stamping that feeds the new-schema
demographics / groups blocks.

These exercise the pure-Python post-pass added so the pipeline emits gender /
age_bucket (from the VLM call already made for staff) and group_id / group_size:
the API's metrics.demographics / metrics.group_stats read exactly these metadata
keys, so getting them stamped is what lights those blocks up.
"""
import datetime as dt

from pipeline.emit import EventWriter, build_event
from pipeline.groups import detect_groups

T0 = dt.datetime(2026, 4, 10, 14, 39, 0, tzinfo=dt.timezone.utc)


def _ev(vid, cam, sec, etype="ZONE_ENTER"):
    return build_event("S", f"CAM_{cam}", vid, etype,
                       T0 + dt.timedelta(seconds=sec), 0.9, zone_id="Z")


def test_groups_pairs_co_arrivers_and_excludes_loner():
    events = []
    for s in range(0, 61, 10):          # A & B arrive ~together, overlap ~60s
        events += [_ev("A", "1", s), _ev("B", "1", s + 5)]
    for s in range(90, 151, 10):        # C alone, much later
        events.append(_ev("C", "1", s))
    g = detect_groups(events, set())
    assert g["A"]["group_id"] == g["B"]["group_id"]   # same group
    assert g["A"]["group_size"] == 2
    assert "C" not in g                               # loner is not a group


def test_groups_excludes_staff_and_drops_floor_wide_cluster():
    events = []
    # six people all co-present on one camera -> bigger than GROUP_MAX -> dropped
    for vid in ("A", "B", "C", "D", "E", "F"):
        for s in range(0, 61, 10):
            events.append(_ev(vid, "1", s))
    assert detect_groups(events, set()) == {}          # too big -> not a group
    # staff are excluded from grouping entirely
    pair = [_ev("A", "1", s) for s in range(0, 61, 10)] + \
           [_ev("S1", "1", s + 5) for s in range(0, 61, 10)]
    assert "S1" not in detect_groups(pair, {"S1"})


def test_stamp_visitor_metadata_writes_only_non_null():
    w = EventWriter()
    w.add(_ev("A", "1", 0))
    w.add(_ev("A", "1", 10))
    w.add(_ev("B", "1", 0))
    touched = w.stamp_visitor_metadata({
        "A": {"gender": "F", "age_bucket": "25-34"},
        "B": {"gender": None, "age_bucket": None},   # all-null -> nothing written
    })
    assert touched == 2                               # both A events
    a = [e for e in w.events() if e.visitor_id == "A"][0]
    b = [e for e in w.events() if e.visitor_id == "B"][0]
    assert a.metadata["gender"] == "F" and a.metadata["age_bucket"] == "25-34"
    assert "gender" not in b.metadata                 # null guess not stamped
