"""Calibration Studio -- a small, SEPARATE web tool (not the graded API).

Why separate: Process 2 (`app/`) is the graded, torch-free API and must stay that
way. The studio is an operator tool that (1) lets you draw zone polygons + the entry
tripwire in the browser on a real frame (no cv2/display needed -- it was the
clunky part of `calibrate.py`), (2) overlays existing zones on the frame so you can
eyeball calibration, and (3) launches the detection pipeline per store as a
background docker job and streams its log.

Run:
    .venv/Scripts/python.exe -m uvicorn studio.app:app --port 8090
    # then open http://localhost:8090

It reads/writes the same per-store config the pipeline uses
(`pipeline/config/...`), so anything you draw here is what the pipeline runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Repo root = the store-intelligence dir (parent of this studio/ package).
ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

# Per-store config + footage registry, loaded from pipeline/config/stores.json so a
# new store appears here just by adding an entry there. Falls back to the two known
# stores if the registry is missing/unreadable (so the studio never hard-fails).
_FALLBACK_STORES: dict[str, dict[str, str]] = {
    "store_1": {"store_id": "STORE_BLR_002", "config_dir": "pipeline/config/store_1",
                "footage": "data/cctv_footage/Store 1", "output": "data/events_store1.jsonl"},
    "store_2": {"store_id": "STORE_MUM_1076", "config_dir": "pipeline/config/store_2",
                "footage": "data/cctv_footage/Store 2", "output": "data/events_store2.jsonl"},
}


def _load_stores() -> dict[str, dict[str, str]]:
    try:
        reg = json.loads((ROOT / "pipeline" / "config" / "stores.json").read_text("utf-8"))
        out: dict[str, dict[str, str]] = {}
        for key, v in (reg.get("stores") or {}).items():
            out[key] = {
                "store_id": v.get("store_id", key),
                "config_dir": v.get("config_dir", f"pipeline/config/{key}"),
                "footage": v.get("footage_dir") or v.get("footage", ""),
                "output": v.get("output", f"data/events_{key}.jsonl"),
            }
        return out or _FALLBACK_STORES
    except Exception:  # noqa: BLE001 - registry optional; never hard-fail the studio
        return _FALLBACK_STORES


STORES: dict[str, dict[str, str]] = _load_stores()

app = FastAPI(title="Store Intelligence — Calibration Studio")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _store(store_key: str) -> dict[str, str]:
    s = STORES.get(store_key)
    if not s:
        raise HTTPException(404, f"unknown store '{store_key}'")
    return s


def _cfg_path(store_key: str, name: str) -> Path:
    return ROOT / _store(store_key)["config_dir"] / name


def _read_json(path: Path, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _frames_dir(store_key: str, clip_path: str) -> Path:
    """Frames were extracted per camera under <footage>/frames/<key>, where key is
    the clip filename before any ' - role' suffix ('CAM 1 - zone' -> 'CAM 1')."""
    stem = Path(clip_path).stem
    key = stem.split(" - ")[0].strip()
    return ROOT / _store(store_key)["footage"] / "frames" / key


def _list_frames(store_key: str, clip_path: str) -> list[str]:
    d = _frames_dir(store_key, clip_path)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


# --------------------------------------------------------------------------- #
# static
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


# --------------------------------------------------------------------------- #
# config / cameras / frames
# --------------------------------------------------------------------------- #
@app.get("/api/stores")
def api_stores() -> dict[str, Any]:
    out = []
    for key, s in STORES.items():
        manifest = _read_json(_cfg_path(key, "clips_manifest.json"), [])
        zones = _read_json(_cfg_path(key, "zones.json"), {})
        cams = []
        for clip in manifest:
            cam = clip["camera_id"]
            frames = _list_frames(key, clip["clip_path"])
            zcfg = zones.get(cam, {})
            cams.append({
                "camera_id": cam,
                "role": clip.get("role"),
                "is_entry": clip.get("role") == "ENTRY",
                "n_frames": len(frames),
                "n_zones": len([z for z in zcfg if isinstance(zcfg.get(z), dict)]),
            })
        out.append({"key": key, "store_id": s["store_id"], "cameras": cams})
    return {"stores": out}


@app.get("/api/frames/{store_key}/{camera_id}")
def api_frames(store_key: str, camera_id: str) -> dict[str, Any]:
    manifest = _read_json(_cfg_path(store_key, "clips_manifest.json"), [])
    clip = next((c for c in manifest if c["camera_id"] == camera_id), None)
    if not clip:
        raise HTTPException(404, "unknown camera")
    return {"frames": _list_frames(store_key, clip["clip_path"])}


@app.get("/api/frame/{store_key}/{camera_id}/{filename}")
def api_frame(store_key: str, camera_id: str, filename: str) -> FileResponse:
    manifest = _read_json(_cfg_path(store_key, "clips_manifest.json"), [])
    clip = next((c for c in manifest if c["camera_id"] == camera_id), None)
    if not clip:
        raise HTTPException(404, "unknown camera")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "bad filename")
    path = _frames_dir(store_key, clip["clip_path"]) / filename
    if not path.is_file():
        raise HTTPException(404, "frame not found")
    return FileResponse(path)


# --------------------------------------------------------------------------- #
# zones + lines (read/write the real per-store config)
# --------------------------------------------------------------------------- #
@app.get("/api/zones/{store_key}")
def api_get_zones(store_key: str) -> dict[str, Any]:
    return {"zones": _read_json(_cfg_path(store_key, "zones.json"), {}),
            "lines": _read_json(_cfg_path(store_key, "lines.json"), {})}


class ZoneBody(BaseModel):
    camera_id: str
    zone_id: str
    polygon: list[list[int]]
    zone_type: Optional[str] = None
    zone_name: Optional[str] = None
    is_revenue_zone: Optional[bool] = None
    is_staff_zone: bool = False
    is_queue_zone: bool = False
    sku_zone: Optional[str] = None


@app.post("/api/zones/{store_key}")
def api_save_zone(store_key: str, body: ZoneBody) -> dict[str, Any]:
    if len(body.polygon) < 3:
        raise HTTPException(400, "a zone needs at least 3 points")
    path = _cfg_path(store_key, "zones.json")
    zones = _read_json(path, {})
    entry: dict[str, Any] = {
        "polygon": body.polygon,
        "is_staff_zone": body.is_staff_zone,
        "is_queue_zone": body.is_queue_zone,
        "sku_zone": body.sku_zone,
    }
    if body.zone_type is not None:
        entry["zone_type"] = body.zone_type
    if body.zone_name is not None:
        entry["zone_name"] = body.zone_name
    if body.is_revenue_zone is not None:
        entry["is_revenue_zone"] = body.is_revenue_zone
    zones.setdefault(body.camera_id, {})[body.zone_id] = entry
    _write_json(path, zones)
    return {"ok": True, "zones": zones.get(body.camera_id, {})}


@app.delete("/api/zones/{store_key}/{camera_id}/{zone_id}")
def api_delete_zone(store_key: str, camera_id: str, zone_id: str) -> dict[str, Any]:
    path = _cfg_path(store_key, "zones.json")
    zones = _read_json(path, {})
    if isinstance(zones.get(camera_id), dict) and zone_id in zones[camera_id]:
        del zones[camera_id][zone_id]
        _write_json(path, zones)
        return {"ok": True}
    raise HTTPException(404, "zone not found")


class LineBody(BaseModel):
    camera_id: str
    p1: list[int]
    p2: list[int]
    inside_point: list[int]
    min_frames_each_side: int = 3


@app.post("/api/lines/{store_key}")
def api_save_line(store_key: str, body: LineBody) -> dict[str, Any]:
    path = _cfg_path(store_key, "lines.json")
    lines = _read_json(path, {})
    lines[body.camera_id] = {
        "p1": body.p1,
        "p2": body.p2,
        "inside_point": body.inside_point,
        "inside_side": "floor",
        "min_frames_each_side": body.min_frames_each_side,
    }
    _write_json(path, lines)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# extract frames from a clip, so there is something to draw zones on
# --------------------------------------------------------------------------- #
@app.post("/api/extract/{store_key}/{camera_id}")
def api_extract(
    store_key: str,
    camera_id: str,
    every_seconds: float = 10.0,
    max_frames: int = 40,
) -> dict[str, Any]:
    """Grab a frame every ``every_seconds`` from the camera's clip into its frames
    dir. Frames are written at the clip's NATIVE resolution, so polygons drawn on
    them line up with what the detection pipeline sees."""
    manifest = _read_json(_cfg_path(store_key, "clips_manifest.json"), [])
    clip = next((c for c in manifest if c["camera_id"] == camera_id), None)
    if not clip:
        raise HTTPException(404, "unknown camera")
    clip_path = ROOT / clip["clip_path"]
    if not clip_path.is_file():
        raise HTTPException(404, f"clip not found at {clip['clip_path']} (put the video there first)")
    try:
        import cv2  # lazy; studio-only optional dep
    except Exception:
        raise HTTPException(
            501,
            "OpenCV is not installed in the studio environment. Run: "
            "pip install -r requirements-studio.txt",
        )

    out_dir = _frames_dir(store_key, clip["clip_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    key = Path(clip["clip_path"]).stem.split(" - ")[0].strip()

    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(fps * max(1.0, every_seconds)))
    written: list[str] = []
    idx = 0
    while len(written) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        name = f"{key}_{int(idx / fps):06d}s.jpg"
        cv2.imwrite(str(out_dir / name), frame)
        written.append(name)
        idx += step
        if total and idx >= total:
            break
    cap.release()
    if not written:
        raise HTTPException(500, "could not decode any frames from the clip")
    frames = sorted(
        p.name for p in out_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return {"ok": True, "written": len(written), "frames": frames}
