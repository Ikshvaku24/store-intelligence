"""Validation renderers for the detection pipeline.

Two modes:

1. DETECT (default) -- runs YOLO+tracker and draws person boxes + track ids + your
   zone polygons + entry tripwire, with live ENTER/EXIT/zone labels. Use it to
   check detection and whether your hand-drawn zones/tripwire line up with reality.

       python -m pipeline.visualize --camera CAM_ENTRY_03 --interval 5

2. OVERLAY (--overlay) -- NO model. Reads the pipeline's --debug-dump JSONL and
   draws each detection coloured by the FINAL decision: red=STAFF, green=customer,
   with the (deduped) visitor_id. This is how you *see* who the VLM/heuristics
   marked as staff and whether dedup collapsed duplicates.

       # 1) produce the dump:
       python -m pipeline.run_pipeline --debug-dump data/debug_tracks.jsonl
       # 2) render it:
       python -m pipeline.visualize --overlay data/debug_tracks.jsonl --out data/cctv_footage/staff_overlay --interval 5

Heavy (cv2; detect mode also needs ultralytics) -- run in the pipeline image/env.
Writes JPGs to <out>/<camera>/ (one every --interval seconds).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

from pipeline import geometry

ZONE_COLOR = (0, 200, 255)
STAFF_ZONE_COLOR = (0, 0, 255)
TRIP_COLOR = (0, 0, 255)
BOX = (0, 255, 0)
STAFF_BOX = (0, 0, 255)
FOOT = (255, 0, 255)


def _load(p: str) -> Any:
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _draw_zones(cv2, np, disp, zone_polys, staff_zone_names):
    for name, poly in zone_polys.items():
        color = STAFF_ZONE_COLOR if name in staff_zone_names else ZONE_COLOR
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(disp, [pts], True, color, 2)
        cv2.putText(disp, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def annotate(camera_id, clip, zones, line, out_dir, interval_s, weights, device, stride):
    import cv2
    import numpy as np

    from pipeline import detect

    os.makedirs(out_dir, exist_ok=True)
    fps = float(clip["fps"])
    role = clip["role"]
    zone_polys = {n: c["polygon"] for n, c in zones.items()}
    staff_zone_names = {n for n, c in zones.items() if c.get("is_staff_zone")}
    tripwire = None
    if line:
        sgn = geometry.inside_sign_from_point(line["p1"], line["p2"], line["inside_point"])
        tripwire = geometry.TripwireCounter(line["p1"], line["p2"], sgn, line.get("min_frames_each_side", 5))

    prev_zone: dict[int, set] = {}
    next_save = 0.0
    saved = 0
    for frame_index, tracks, frame in detect.track_clip(
        clip["clip_path"], weights=weights, vid_stride=stride, device=device
    ):
        if frame is None:
            continue
        t = frame_index / fps
        for tr in tracks:
            fp = tr.foot_point
            zs = list(geometry.zones_for_point(zone_polys, fp))
            ev = ""
            if tripwire is not None:
                c = tripwire.update(tr.track_id, fp)
                if c:
                    ev = c
            prev = prev_zone.setdefault(tr.track_id, set())
            new = [z for z in zs if z not in prev]
            prev_zone[tr.track_id] = set(zs)
            if new and not ev:
                ev = "ENTER " + ",".join(new)
            tr._zs = zs  # type: ignore[attr-defined]
            tr._ev = ev  # type: ignore[attr-defined]
        if t < next_save:
            continue
        next_save = t + interval_s
        disp = frame.copy()
        _draw_zones(cv2, np, disp, zone_polys, staff_zone_names)
        if tripwire is not None and line:
            cv2.line(disp, tuple(line["p1"]), tuple(line["p2"]), TRIP_COLOR, 3)
            cv2.circle(disp, tuple(line["inside_point"]), 6, TRIP_COLOR, -1)
            cv2.putText(disp, "INSIDE", tuple(line["inside_point"]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TRIP_COLOR, 2)
        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            cv2.rectangle(disp, (x1, y1), (x2, y2), BOX, 2)
            fx, fy = [int(v) for v in tr.foot_point]
            cv2.circle(disp, (fx, fy), 5, FOOT, -1)
            zs = getattr(tr, "_zs", [])
            ev = getattr(tr, "_ev", "")
            lab = f"T{tr.track_id} {tr.confidence:.2f}"
            if zs:
                lab += " [" + ",".join(zs) + "]"
            cv2.putText(disp, lab, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX, 2)
            if ev:
                cv2.putText(disp, ev, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TRIP_COLOR, 2)
        cv2.putText(disp, f"{camera_id} {role} t={t:5.1f}s n={len(tracks)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(out_dir, f"{camera_id}_{int(t):06d}s.jpg"), disp)
        saved += 1
    print(f"[{camera_id}] saved {saved} annotated frames to {out_dir}")


def overlay(camera_id, clip, zones, rows_for_cam, out_dir, interval_s):
    """Draw the pipeline's final decisions (staff/customer) from a debug dump."""
    import cv2
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    fps = float(clip["fps"])
    role = clip["role"]
    zone_polys = {n: c["polygon"] for n, c in zones.items()}
    staff_zone_names = {n for n, c in zones.items() if c.get("is_staff_zone")}

    by_fi: dict[int, list] = defaultdict(list)
    for r in rows_for_cam:
        by_fi[r["frame_index"]].append(r)

    cap = cv2.VideoCapture(clip["clip_path"])
    if not cap.isOpened():
        print(f"[{camera_id}] cannot open {clip['clip_path']}")
        return
    saved = 0
    next_save = 0.0
    for fi in sorted(by_fi):
        t = fi / fps
        if t < next_save:
            continue
        next_save = t + interval_s
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, disp = cap.read()
        if not ok:
            continue
        _draw_zones(cv2, np, disp, zone_polys, staff_zone_names)
        nstaff = 0
        for r in by_fi[fi]:
            x1, y1, x2, y2 = r["bbox"]
            staff = bool(r.get("is_staff"))
            nstaff += staff
            color = STAFF_BOX if staff else BOX
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            lab = ("STAFF " if staff else "CUST ") + str(r.get("visitor_id", ""))[-6:]
            cv2.putText(disp, lab, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(disp, f"{camera_id} {role} t={t:5.1f}s people={len(by_fi[fi])} staff={nstaff}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(out_dir, f"{camera_id}_{int(t):06d}s.jpg"), disp)
        saved += 1
    cap.release()
    print(f"[{camera_id}] saved {saved} overlay frames to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pipeline/config/store_1/clips_manifest.json")
    ap.add_argument("--zones", default="pipeline/config/store_1/zones.json")
    ap.add_argument("--lines", default="pipeline/config/store_1/lines.json")
    ap.add_argument("--camera", default=None, help="camera_id to render (default: all)")
    ap.add_argument("--interval", type=float, default=10.0, help="seconds between saved frames")
    ap.add_argument("--out", default="data/cctv_footage/annotated")
    ap.add_argument("--overlay", default=None,
                    help="pipeline --debug-dump JSONL; draw staff/customer (no model needed)")
    ap.add_argument("--weights", default=os.environ.get("WEIGHTS", "yolo11m.pt"))
    ap.add_argument("--stride", type=int, default=int(os.environ.get("VID_STRIDE", "3")))
    ap.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    a = ap.parse_args()

    manifest = _load(a.manifest)
    zones = _load(a.zones)
    lines = _load(a.lines)

    if a.overlay:
        rows = [json.loads(line) for line in open(a.overlay, encoding="utf-8") if line.strip()]
        by_cam: dict[str, list] = defaultdict(list)
        for r in rows:
            by_cam[r["camera"]].append(r)
        for clip in manifest:
            cam = clip["camera_id"]
            if a.camera and cam != a.camera:
                continue
            overlay(cam, clip, zones.get(cam, {}), by_cam.get(cam, []),
                    os.path.join(a.out, cam), a.interval)
        return

    for clip in manifest:
        cam = clip["camera_id"]
        if a.camera and cam != a.camera:
            continue
        annotate(cam, clip, zones.get(cam, {}), lines.get(cam), os.path.join(a.out, cam),
                 a.interval, a.weights, a.device, a.stride)


if __name__ == "__main__":
    main()
