"""Appearance features for identity only (BUILD_SPEC Section 7.3).

(a) a CNN appearance embedding (L2-normalised) and (b) an HSV colour histogram as a
cheap fallback. These feed re-entry, best-effort cross-camera linking, and the
post-run visitor dedup (pipeline/dedup.py) -- NEVER entry counting.

Embedding backend: **torchvision MobileNetV3-small** (pretrained, classifier
stripped -> 576-d pooled features). We moved off ``torchreid``/OSNet because
torchreid 0.2.5 imports the removed ``torch._six`` and so fails to initialise on
torch 2.x -- it silently produced *no* embeddings, which disabled dedup entirely
(visitor over-count). torchvision is already a dependency and works on torch 2.x.

Everything is imported lazily and degrades gracefully: if torch/torchvision are
unavailable the extractor returns None embeddings (orchestrator still runs, just
without appearance dedup). The chosen backend is printed once so a run shows it.
"""
from __future__ import annotations

from typing import Optional


class FeatureExtractor:
    def __init__(self, model_name: str = "mobilenet_v3_small", device: Optional[str] = None):
        self.device = self._resolve_device(device)
        self._model = None
        self._preprocess = None
        self._torch = None
        self.backend = "none"
        try:
            import torch  # lazy
            from torchvision.models import (  # lazy
                mobilenet_v3_small,
                MobileNet_V3_Small_Weights,
            )

            weights = MobileNet_V3_Small_Weights.DEFAULT
            model = mobilenet_v3_small(weights=weights)
            model.classifier = torch.nn.Identity()  # keep the 576-d pooled features
            model.eval().to(self.device)
            self._model = model
            self._preprocess = weights.transforms()
            self._torch = torch
            self.backend = model_name
        except Exception as exc:  # noqa: BLE001 - degrade to histogram/none
            print(f"[reid] WARNING: appearance embedder unavailable ({exc}); "
                  f"dedup will be disabled.", flush=True)
            self._model = None
        print(f"[reid] embedding backend: {self.backend} (device {self.device})", flush=True)

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device not in (None, "", "auto"):
            return device
        try:
            import torch  # lazy

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def embed(self, crop) -> Optional[list[float]]:
        """Appearance embedding for an image crop (BGR ndarray). None if unavailable."""
        if self._model is None:
            return None
        try:
            import numpy as np
            from PIL import Image

            rgb = np.ascontiguousarray(crop[:, :, ::-1])  # BGR -> RGB
            tensor = self._preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            with self._torch.no_grad():
                feat = self._model(tensor)[0]
            return _l2_normalize(feat.detach().cpu().tolist())
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def hsv_histogram(crop, bins: int = 16) -> Optional[list[float]]:
        """Normalised HSV hue/sat histogram for a crop. None if cv2 unavailable."""
        try:
            import cv2  # lazy
            import numpy as np

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist.astype(float).tolist()
        except Exception:  # noqa: BLE001
            return None


def crop_bbox(frame, bbox) -> Optional["object"]:
    """Crop an (x1,y1,x2,y2) bbox from a frame ndarray, clamped to bounds."""
    try:
        import numpy as np  # noqa: F401

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]
    except Exception:  # noqa: BLE001
        return None


def encode_jpeg(crop, max_w: int = 256) -> Optional[bytes]:
    """Downscale a BGR crop to <= max_w wide and return JPEG bytes (for the VLM).

    None if cv2 is unavailable or the crop is empty."""
    try:
        import cv2  # lazy

        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        h, w = crop.shape[:2]
        if w > max_w:
            scale = max_w / float(w)
            crop = cv2.resize(crop, (max_w, max(1, int(h * scale))))
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None
    except Exception:  # noqa: BLE001
        return None


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return vec
    return [v / norm for v in vec]
