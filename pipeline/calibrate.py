"""Hand-draw zone polygons and the entry tripwire on a sample frame
(BUILD_SPEC Section 5). Calibration replaces training data.

Usage (interactive, needs a display + opencv):
    python pipeline/calibrate.py --camera CAM_ENTRY_03 --frame data/cam_3_frame.png --mode line
    python pipeline/calibrate.py --camera CAM_SKIN_01 --frame data/cam_1_frame.png --mode zone --zone SKINCARE_MOISTURISER

Left-click adds points; press 's' to save, 'n' for the next polygon, 'q' to quit.
Results are merged into pipeline/config/zones.json or lines.json. Coordinates are
on the distorted frame, so no lens calibration is required.

A frame can be extracted from a clip with:
    ffmpeg -i data/cam_3_entry.mp4 -vframes:1 data/cam_3_frame.png
"""
from __future__ import annotations

import argparse
import json
import os

ZONES_PATH = "pipeline/config/store_1/zones.json"
LINES_PATH = "pipeline/config/store_1/lines.json"


def _load(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"Saved -> {path}")


def collect_points(frame_path: str, title: str) -> list[list[int]]:
    import cv2

    img = cv2.imread(frame_path)
    if img is None:
        raise SystemExit(f"Could not read frame: {frame_path}")

    h, w = img.shape[:2]

    MAX_W = 1400
    MAX_H = 800

    scale = min(MAX_W / w, MAX_H / h)

    disp_w = int(w * scale)
    disp_h = int(h * scale)

    display_points: list[list[int]] = []

    win = f"calibrate: {title} (click points, 's' save, 'u' undo, 'q' quit)"

    disp_img = cv2.resize(
        img,
        (disp_w, disp_h),
        interpolation=cv2.INTER_AREA,
    )

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            display_points.append([x, y])

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = disp_img.copy()

        for i, p in enumerate(display_points):
            cv2.circle(disp, tuple(p), 5, (0, 255, 0), -1)

            if i > 0:
                cv2.line(
                    disp,
                    tuple(display_points[i - 1]),
                    tuple(p),
                    (0, 255, 0),
                    2,
                )

        cv2.imshow(win, disp)

        key = cv2.waitKey(20) & 0xFF

        if key == ord("u"):
            if display_points:
                display_points.pop()

        elif key in (ord("s"), ord("q")):
            break

    cv2.destroyAllWindows()

    points = [
        [int(x / scale), int(y / scale)]
        for x, y in display_points
    ]

    return points


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--mode", choices=["zone", "line"], required=True)
    ap.add_argument("--zone", help="zone name (mode=zone)")
    ap.add_argument("--staff-zone", action="store_true")
    ap.add_argument("--queue-zone", action="store_true")
    ap.add_argument("--sku-zone", default=None)
    # New-schema zone attributes (carried into event metadata; match the graders'
    # zone_type / zone_name / is_revenue_zone fields). Optional -> omitted if unset.
    ap.add_argument("--zone-type", default=None, help="SHELF | DISPLAY | BILLING | ...")
    ap.add_argument("--zone-name", default=None, help="human label, e.g. 'Left Wall Shelf'")
    ap.add_argument("--revenue", dest="revenue", action="store_true", help="mark is_revenue_zone=true")
    # Per-store config files (default to the single top-level config = Store 1).
    ap.add_argument("--zones-path", default=ZONES_PATH)
    ap.add_argument("--lines-path", default=LINES_PATH)
    args = ap.parse_args()

    if args.mode == "zone":
        if not args.zone:
            raise SystemExit("--zone is required for mode=zone")
        pts = collect_points(args.frame, args.zone)
        cfg = _load(args.zones_path)
        entry = {
            "polygon": pts,
            "is_staff_zone": bool(args.staff_zone),
            "is_queue_zone": bool(args.queue_zone),
            "sku_zone": args.sku_zone,
        }
        if args.zone_type is not None:
            entry["zone_type"] = args.zone_type
        if args.zone_name is not None:
            entry["zone_name"] = args.zone_name
        if args.revenue:
            entry["is_revenue_zone"] = True
        cfg.setdefault(args.camera, {})[args.zone] = entry
        _save(args.zones_path, cfg)
    else:
        pts = collect_points(args.frame, "tripwire (click 2 endpoints + 1 inside point)")
        if len(pts) < 3:
            raise SystemExit("Need 3 clicks: p1, p2, then a point inside the store.")
        cfg = _load(args.lines_path)
        cfg[args.camera] = {
            "p1": pts[0],
            "p2": pts[1],
            "inside_point": pts[2],
            "inside_side": "floor",
            "min_frames_each_side": 3,
        }
        _save(args.lines_path, cfg)


if __name__ == "__main__":
    main()
