"""Decode clips -> YOLO person detect -> ByteTrack tracks (BUILD_SPEC Section 7.2).

Heavy CV deps (ultralytics/torch) are imported lazily inside the functions so the
module can be imported in the lightweight API venv (e.g. by tests that only touch
the pure-Python geometry/session logic).

Per frame we yield {track_id, confidence, bbox, foot_point}. An empty frame
yields an empty list -- a valid 'empty store' state, never an error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

# A deliberately low confidence floor so ByteTrack's second association stage can
# rescue low-score boxes through occlusion (BUILD_SPEC Section 7.2).
DEFAULT_CONF = 0.20
PERSON_CLASS = 0  # COCO 'person'


@dataclass
class Track:
    track_id: int
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def foot_point(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)  # bottom-centre


def load_model(weights: str = "yolo11m.pt"):
    from ultralytics import YOLO  # lazy

    return YOLO(weights)


def track_clip(
    clip_path: str,
    weights: str = "yolo11m.pt",
    conf: float = DEFAULT_CONF,
    tracker: str = "bytetrack.yaml",
    track_buffer_frames: Optional[int] = None,
    imgsz: int = 1280,
    with_frames: bool = True,
    vid_stride: int = 1,
    device: Optional[str] = None,
) -> Iterator[tuple[int, list[Track], object]]:
    """Yield (frame_index, tracks, frame_bgr) for every frame of the clip.

    frame_bgr is the original image (for Re-ID crops) or None if with_frames is
    False. vid_stride processes every Nth frame (a big CPU speed-up); the yielded
    frame_index is the TRUE frame number so event timestamps stay correct.
    track_buffer_frames enlarges ByteTrack's buffer so a person passing behind a
    standee/display survives the gap and keeps the same track id.
    """
    model = load_model(weights)
    # ultralytics wants None for auto-select (GPU if available), or 'cpu' / 0 / 'cuda:0'.
    dev = None if device in (None, "", "auto") else device
    stream = model.track(
        source=clip_path,
        stream=True,
        persist=True,
        classes=[PERSON_CLASS],
        conf=conf,
        tracker=tracker,
        imgsz=imgsz,
        vid_stride=vid_stride,
        device=dev,
        verbose=False,
    )
    for enum_index, result in enumerate(stream):
        frame_index = enum_index * vid_stride
        tracks: list[Track] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.tolist()
            ids = boxes.id.int().tolist()
            confs = boxes.conf.tolist()
            for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, confs):
                tracks.append(Track(track_id=int(tid), confidence=float(c), bbox=(x1, y1, x2, y2)))
        frame = getattr(result, "orig_img", None) if with_frames else None
        yield frame_index, tracks, frame
