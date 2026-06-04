"""Conservative group-shopping detection (new-schema ``group_id`` / ``group_size``).

People who shop TOGETHER tend to (a) appear on the same camera and (b) arrive at
roughly the same time and overlap in dwell. We form groups from non-staff *canonical*
visitors (run after dedup, so one id ~= one person) whose presence intervals on a
shared camera overlap for >= ``MIN_OVERLAP_S`` AND start within ``ENTER_GAP_S`` of each
other. The arrival-together cue is what keeps random co-shoppers on a busy floor out
of a group. Union-find merges transitive pairs; only groups of size
``GROUP_MIN..GROUP_MAX`` are kept -- a "group" spanning the whole floor is noise and is
dropped (so a bad run degrades to *no* groups, never a giant fake one).

Approximate on crowded / fragmented footage -> flagged in DESIGN/CHOICES. Pure-Python
(reads only visitor_id / camera_id / timestamp), so it is unit-testable in the light
venv with no cv2/torch.
"""
from __future__ import annotations

from typing import Any

ENTER_GAP_S = 15.0      # the two visitors were first seen within this many seconds
MIN_OVERLAP_S = 20.0    # ...and share at least this much presence on one camera
GROUP_MIN = 2
GROUP_MAX = 4


def detect_groups(events: list[Any], staff_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Return ``{visitor_id: {"group_id", "group_size"}}`` for grouped visitors.

    ``events`` are StoreEvent-like objects exposing ``visitor_id`` / ``camera_id`` /
    ``timestamp`` (datetime). Staff are excluded.
    """
    # presence interval per (visitor, camera)
    span: dict[tuple[str, str], list] = {}
    for e in events:
        vid = e.visitor_id
        if vid in staff_ids:
            continue
        key = (vid, e.camera_id)
        ts = e.timestamp
        s = span.get(key)
        if s is None:
            span[key] = [ts, ts]
        else:
            if ts < s[0]:
                s[0] = ts
            if ts > s[1]:
                s[1] = ts

    by_cam: dict[str, list] = {}
    for (vid, cam), (first, last) in span.items():
        by_cam.setdefault(cam, []).append((vid, first, last))

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for lst in by_cam.values():
        for i in range(len(lst)):
            vi, fi, li = lst[i]
            for j in range(i + 1, len(lst)):
                vj, fj, lj = lst[j]
                if vi == vj:
                    continue
                overlap = (min(li, lj) - max(fi, fj)).total_seconds()
                gap = abs((fi - fj).total_seconds())
                if overlap >= MIN_OVERLAP_S and gap <= ENTER_GAP_S:
                    union(vi, vj)

    members: dict[str, set] = {}
    for vid in {v for (v, _cam) in span}:
        members.setdefault(find(vid), set()).add(vid)

    out: dict[str, dict[str, Any]] = {}
    gi = 0
    for mem in members.values():
        if not (GROUP_MIN <= len(mem) <= GROUP_MAX):
            continue  # singletons and floor-wide clusters are not groups
        gi += 1
        gid = f"G{gi:02d}"
        for vid in mem:
            out[vid] = {"group_id": gid, "group_size": len(mem)}
    return out
