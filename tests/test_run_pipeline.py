# PROMPT: "Write a pytest test that drives the whole detection orchestrator
#   (pipeline/run_pipeline.run) end-to-end with the heavy detector mocked out:
#   feed synthetic per-frame tracks that cross the entry tripwire and dwell in a
#   floor zone, then assert events.jsonl is produced with a valid ENTRY and a
#   ZONE_ENTER, all conforming to the StoreEvent schema."
# CHANGES MADE: The AI mocked detection but forgot the orchestrator imports
#   FeatureExtractor/crop_bbox lazily, so I patched those module attributes (not a
#   constructor arg). I authored a tiny temp manifest/zones/lines with known
#   geometry so the crossing is deterministic, and added the assertion that every
#   emitted line re-validates through StoreEvent (the pipeline->API contract).
from __future__ import annotations

import json

from pipeline import detect, reid, run_pipeline
from pipeline.detect import Track
from app.models import StoreEvent


class _StubExtractor:
    def __init__(self, device=None):
        self.device = "cpu"

    def embed(self, crop):
        return None

    @staticmethod
    def hsv_histogram(crop, bins=16):
        return None


def _fake_track_clip(clip_path, weights="yolo11m.pt", **kw):
    # foot_point = ((x1+x2)/2, y2). Outside (y=50) for 3 frames, then inside
    # (y=150) for the rest -> a clean inbound crossing of the y=100 tripwire and
    # presence inside the floor zone.
    def track_at(y):
        return Track(track_id=1, confidence=0.8, bbox=(90.0, y - 100.0, 110.0, float(y)))

    seq = [50, 50, 50, 150, 150, 150, 150, 150]
    for i, y in enumerate(seq):
        yield i, [track_at(y)], None


def _write(tmp, name, obj):
    p = tmp / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def test_orchestrator_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(detect, "track_clip", _fake_track_clip)
    monkeypatch.setattr(reid, "FeatureExtractor", _StubExtractor)
    monkeypatch.setattr(reid, "crop_bbox", lambda frame, bbox: None)

    manifest = _write(
        tmp_path,
        "manifest.json",
        [
            {"clip_path": "entry.mp4", "store_id": "STORE_X", "camera_id": "CAM_ENTRY_03", "role": "ENTRY", "start_ts": "2026-04-16T08:00:00Z", "fps": 15},
            {"clip_path": "floor.mp4", "store_id": "STORE_X", "camera_id": "CAM_SKIN_01", "role": "FLOOR_SKINCARE", "start_ts": "2026-04-16T08:00:00Z", "fps": 15},
        ],
    )
    zones = _write(
        tmp_path,
        "zones.json",
        {
            "CAM_ENTRY_03": {},
            "CAM_SKIN_01": {
                "SKINCARE_MOISTURISER": {
                    "polygon": [[0, 100], [200, 100], [200, 300], [0, 300]],
                    "is_staff_zone": False,
                    "is_queue_zone": False,
                    "sku_zone": "SKINCARE_MOISTURISER",
                }
            },
        },
    )
    lines = _write(
        tmp_path,
        "lines.json",
        {"CAM_ENTRY_03": {"p1": [0, 100], "p2": [200, 100], "inside_point": [100, 200], "min_frames_each_side": 2}},
    )
    out = str(tmp_path / "events.jsonl")

    n = run_pipeline.run(manifest, zones, lines, out, weights="none")
    assert n > 0

    with open(out, encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh if line.strip()]

    types = {e["event_type"] for e in events}
    assert "ENTRY" in types
    assert "ZONE_ENTER" in types

    # Every emitted event must re-validate against the shared schema.
    for e in events:
        StoreEvent.model_validate(e)

    # Entry events carry no zone_id; zone events do.
    entry = next(e for e in events if e["event_type"] == "ENTRY")
    assert entry["zone_id"] is None
    zone = next(e for e in events if e["event_type"] == "ZONE_ENTER")
    assert zone["zone_id"] == "SKINCARE_MOISTURISER"
    assert zone["metadata"]["sku_zone"] == "SKINCARE_MOISTURISER"
