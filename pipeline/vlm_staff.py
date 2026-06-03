"""Gemini VLM staff/customer classifier (BUILD_SPEC Part D — AI usage).

Behavioural role classification for the cases geometry can't see: a person
applying makeup to someone, demonstrating/handing products, operating the POS, or
restocking. This is the deliberate, documented adoption of a VLM for staff
detection (overriding the earlier position-only stance) precisely because this
footage has **no uniform** and a busy floor where position is silent.

Design:
* one cached verdict per ``visitor_id`` (cap one call per person; reruns are free
  and deterministic from the cache);
* graceful degradation — if the SDK or API key is absent, ``available`` is False
  and the resolver falls back to position + heuristic;
* the exact prompt lives here so it can be quoted verbatim in DESIGN.md/CHOICES.md.

Free-tier safety: Gemini free tiers have a low requests-per-minute cap (e.g.
~10 RPM for gemini-2.5-flash). The resolver calls this once per ambiguous visitor,
which can burst dozens of calls. To avoid 429 RESOURCE_EXHAUSTED storms we
(a) throttle to a minimum gap between calls, (b) retry 429s with backoff instead
of silently dropping the verdict, and (c) cap total calls per run.

Env:
    GEMINI_API_KEY (or GOOGLE_API_KEY)   API key
    GEMINI_MODEL                          model id (default gemini-2.5-flash)
    VLM_STAFF_CACHE                       cache path (default data/vlm_staff_cache.json)
    VLM_MIN_INTERVAL_S                    min seconds between calls (default 6.5 -> ~9 RPM)
    VLM_MAX_CALLS                         hard cap on calls per run (default 50; 0 = unlimited)
    VLM_MAX_RETRIES                       retries on a 429 before giving up (default 4)
"""
from __future__ import annotations

import json
import os
import time
from io import BytesIO
from typing import Any, Optional

PROMPT = (
    "You are analysing still CCTV frames from a cosmetics store. Faces are blurred. "
    "Decide if this ONE person is STAFF or a CUSTOMER.\n"
    "DEFAULT TO CUSTOMER: the large majority of people in a store are customers. "
    "Only answer STAFF when there is CLEAR, REPEATED evidence of a working role -- "
    "when unsure, answer CUSTOMER.\n\n"
    "STAFF (needs clear evidence): wears a consistent store uniform/apron; stands "
    "BEHIND the billing counter operating the POS/scanner; applies makeup to or "
    "serves several DIFFERENT people; restocks shelves from cartons; repeatedly "
    "assists different customers across the frames.\n"
    "CUSTOMER (default): browses or picks products for themselves; carries a shopping "
    "bag/handbag; is the one being served / having makeup applied; or the role is "
    "unclear.\n\n"
    "The frames are crops of the SAME person over time. Context: {context}.\n"
    "Set confidence > 0.8 ONLY when the staff role is obvious; use low confidence "
    "when unsure. Respond with STRICT JSON only, no prose: "
    '{{"is_staff": true|false, "confidence": 0.0-1.0, "reason": "<short>"}}'
)


class GeminiStaffClassifier:
    """Cached Gemini classifier. ``available`` is False if SDK/key missing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_path: Optional[str] = None,
    ) -> None:
        self.model_name = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.cache_path = cache_path or os.environ.get("VLM_STAFF_CACHE", "data/vlm_staff_cache.json")
        self._cache: dict[str, Any] = self._load_cache()
        self._client = None
        self.available = False

        # Free-tier rate-limit controls.
        self.min_interval = float(os.environ.get("VLM_MIN_INTERVAL_S", "0.5"))
        self.max_calls = int(os.environ.get("VLM_MAX_CALLS", "50"))
        self.max_retries = int(os.environ.get("VLM_MAX_RETRIES", "4"))
        self._calls = 0
        self._last_call = 0.0

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            return
        try:
            from google import genai  # lazy (google-genai SDK)

            self._client = genai.Client(api_key=key)
            self.available = True
        except Exception:  # noqa: BLE001 - any import/config failure -> degrade
            self.available = False

    # ------------------------------------------------------------------ #
    def classify(
        self, visitor_id: str, crops_jpeg: list[bytes], context: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Return {is_staff, confidence, reason, source} or None if unavailable.

        Cached per visitor_id (including negative/customer verdicts)."""
        if visitor_id in self._cache:
            return self._cache[visitor_id]
        if not self.available or not crops_jpeg:
            return None
        if self.max_calls and self._calls >= self.max_calls:
            return None  # per-run budget exhausted; degrade to position+heuristic

        try:
            from PIL import Image  # lazy

            images = [Image.open(BytesIO(b)).convert("RGB") for b in crops_jpeg[:6]]
            prompt = PROMPT.format(context=json.dumps(context))
            text = self._generate(prompt, images)
            verdict = _parse_verdict(text or "")
        except Exception:  # noqa: BLE001 - network/parse failure -> no verdict
            return None

        if verdict is None:
            return None
        verdict["source"] = "vlm"
        self._cache[visitor_id] = verdict
        self._save_cache()
        return verdict

    # ------------------------------------------------------------------ #
    def _generate(self, prompt: str, images: list[Any]) -> Optional[str]:
        """Call Gemini with free-tier throttling + 429 backoff. Returns reply text.

        Raises on a non-rate-limit error (caught by ``classify``); returns None if
        the rate limit can't be cleared within ``max_retries``."""
        for attempt in range(self.max_retries + 1):
            # Throttle: keep at least ``min_interval`` seconds between calls.
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                self._calls += 1
                self._last_call = time.monotonic()
                print(f"  [vlm] call {self._calls}"
                      f"{f'/{self.max_calls}' if self.max_calls else ''} "
                      f"(model {self.model_name})...", flush=True)
                resp = self._client.models.generate_content(
                    model=self.model_name, contents=[prompt, *images]
                )
                return getattr(resp, "text", "") or ""
            except Exception as exc:  # noqa: BLE001
                if not _is_rate_limit(exc) or attempt >= self.max_retries:
                    if _is_rate_limit(exc):
                        return None  # give up on this visitor, keep the run going
                    raise
                # Exponential backoff, honouring a server-suggested retryDelay.
                delay = _retry_delay(exc) or min(60.0, 5.0 * (2 ** attempt))
                time.sleep(delay)
        return None

    # ------------------------------------------------------------------ #
    def _load_cache(self) -> dict[str, Any]:
        try:
            with open(self.cache_path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as fh:
                json.dump(self._cache, fh, indent=2)
        except OSError:
            pass


def _is_rate_limit(exc: Exception) -> bool:
    """True if the exception is a 429 / quota-exhausted error from any SDK shape."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    msg = str(exc).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "RATE LIMIT" in msg


def _retry_delay(exc: Exception) -> Optional[float]:
    """Extract a server-suggested retry delay (e.g. 'retryDelay': '37s') if present."""
    import re

    m = re.search(r"retry[-_ ]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", str(exc), re.I)
    if m:
        try:
            return min(60.0, float(m.group(1)))
        except ValueError:
            return None
    return None


def _parse_verdict(text: str) -> Optional[dict[str, Any]]:
    """Extract the JSON object from a model reply, tolerating ``` fences."""
    text = text.strip()
    if "{" in text and "}" in text:
        text = text[text.index("{"): text.rindex("}") + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if "is_staff" not in obj:
        return None
    return {
        "is_staff": bool(obj.get("is_staff")),
        "confidence": float(obj.get("confidence", 0.0) or 0.0),
        "reason": str(obj.get("reason", ""))[:200],
    }
