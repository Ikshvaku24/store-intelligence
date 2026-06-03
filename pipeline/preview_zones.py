"""Draw zones.json polygons + lines.json tripwire on STATIC frames (no model).

A fast sanity check for calibration: overlays every zone (filled + named) and the
entry tripwire on the extracted frame, so you can see if the polygons sit on the
right shelves before spending GPU time on a full run.

    python -m pipeline.preview_zones                  # all mapped frames
    python -m pipeline.preview_zones --frame data/cam1_frame.png --camera CAM_SKIN_01

Writes PNGs to data/zone_preview/<camera>.png. cv2 only (works in the API venv).
Polygons are in 1920x1080 video space; pass frames extracted at that resolution.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

# frame file -> camera_id in zones.json (edit if your frames are named differently)
DEFAULT_MAP = {
    "data/cam1_frame.png": "CAM_SKIN_01",
    "data/cam2_frame.png": "CAM_MAKEUP_02",
    "data/cam3_frame.png": "CAM_ENTRY_03",
    "data/cam4_frame.png": "CAM_BACK_04",
    "data/cam5_frame.png": "CAM_BILL_05",
}


def _color(cfg: dict) -> tuple:
    if cfg.get("is_staff_zone"):
        return (0, 0, 255)        # red = staff zone
    if cfg.get("is_queue_zone"):
        return (255, 128, 0)      # blue = queue zone
    return (0, 200, 255)          # amber = product zone


def render(frame_path: str, camera_id: str, zones: dict, lines: dict, out_dir: str) -> None:
    import cv2
    import numpy as np

    img = cv2.imread(frame_path)
    if img is None:
        print(f"  skip (no frame): {frame_path}")
        return
    z = zones.get(camera_id, {})
    for name, cfg in z.items():
        poly = cfg["polygon"]
        color = _color(cfg)
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        cv2.polylines(img, [pts], True, color, 2)
        cx = int(sum(p[0] for p in poly) / len(poly))
        cy = int(sum(p[1] for p in poly) / len(poly))
        cv2.putText(img, name, (cx - 70, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    ln = lines.get(camera_id)
    if ln:
        cv2.line(img, tuple(ln["p1"]), tuple(ln["p2"]), (0, 0, 255), 3)
        cv2.circle(img, tuple(ln["inside_point"]), 8, (0, 0, 255), -1)
        cv2.putText(img, "TRIPWIRE", tuple(ln["p1"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{camera_id}.png")
    cv2.imwrite(out, img)
    print(f"  wrote {out}  zones={list(z)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", default="pipeline/config/store_1/zones.json")
    ap.add_argument("--lines", default="pipeline/config/store_1/lines.json")
    ap.add_argument("--out", default="data/zone_preview")
    ap.add_argument("--frame", default=None)
    ap.add_argument("--camera", default=None)
    a = ap.parse_args()

    with open(a.zones, encoding="utf-8") as fh:
        zones = json.load(fh)
    try:
        with open(a.lines, encoding="utf-8") as fh:
            lines = json.load(fh)
    except (OSError, json.JSONDecodeError):
        lines = {}

    if a.frame and a.camera:
        render(a.frame, a.camera, zones, lines, a.out)
        return
    for frame_path, cam in DEFAULT_MAP.items():
        render(frame_path, cam, zones, lines, a.out)


if __name__ == "__main__":
    main()
