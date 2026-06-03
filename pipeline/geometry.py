"""Tripwire line-crossing + point-in-polygon zone tests (BUILD_SPEC Section 7.4).

Pure Python (no shapely) so the geometry -- the single most reliable signal in
this pipeline -- is dependency-light and unit-testable in the API venv. The math
is identical to shapely's for these primitives: a cross-product sign for which
side of the tripwire a foot-point is on, and ray-casting for zone membership.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

Point = Sequence[float]


def cross_sign(p1: Point, p2: Point, pt: Point) -> int:
    """Sign of the cross product (p2-p1) x (pt-p1): which side of line p1->p2."""
    val = (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])
    if val > 1e-9:
        return 1
    if val < -1e-9:
        return -1
    return 0


def point_in_polygon(polygon: Sequence[Point], pt: Point) -> bool:
    """Ray-casting point-in-polygon. Works on the distorted frame directly (no
    lens calibration needed -- polygons are drawn on the distorted image)."""
    x, y = pt[0], pt[1]
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


class TripwireCounter:
    """Debounced directional line-crossing detector.

    A crossing is only confirmed once a track has been observed on a side for
    ``min_frames`` consecutive frames (the brief's K-frames-each-side debounce),
    which rejects jitter and motion-blur flicker at the dark marble threshold.
    Direction is resolved against ``inside_sign`` (the sign of a known inside
    point) -> ENTRY if the new stable side is inside, else EXIT.
    """

    def __init__(self, p1: Point, p2: Point, inside_sign: int, min_frames: int = 3):
        self.p1 = p1
        self.p2 = p2
        self.inside_sign = inside_sign
        self.min_frames = max(1, min_frames)
        self._state: dict[int, dict] = {}

    def update(self, track_id: int, foot_point: Point) -> Optional[str]:
        side = cross_sign(self.p1, self.p2, foot_point)
        if side == 0:
            return None  # exactly on the line -- wait for a decisive frame

        st = self._state.setdefault(
            track_id, {"cur_side": side, "cur_count": 0, "stable_side": None}
        )
        if side == st["cur_side"]:
            st["cur_count"] += 1
        else:
            st["cur_side"] = side
            st["cur_count"] = 1

        if st["cur_count"] >= self.min_frames and side != st["stable_side"]:
            previous = st["stable_side"]
            st["stable_side"] = side
            if previous is not None:
                return "ENTRY" if side == self.inside_sign else "EXIT"
        return None


def inside_sign_from_point(p1: Point, p2: Point, inside_point: Point) -> int:
    """Convert a human-friendly 'a point known to be inside the store' into the
    cross-product sign the TripwireCounter compares against."""
    s = cross_sign(p1, p2, inside_point)
    return s if s != 0 else 1


def zones_for_point(zones: dict[str, Sequence[Point]], pt: Point) -> Iterable[str]:
    """Yield names of every zone polygon containing ``pt``."""
    for name, poly in zones.items():
        if point_in_polygon(poly, pt):
            yield name
