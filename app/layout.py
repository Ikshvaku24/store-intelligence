"""Loader for store_layout.json (open hours + canonical zone names).

Open hours let the API distinguish "empty during open hours" (a DEAD_ZONE
anomaly) from "closed" (expected silence). Missing/!absent layout degrades
gracefully: open hours default to always-open and zone names are inferred.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from functools import lru_cache
from typing import Any, Optional

from app.config import get_settings


@lru_cache
def load_layout() -> dict[str, Any]:
    path = get_settings().store_layout_path
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _store_block(store_id: str) -> dict[str, Any]:
    layout = load_layout()
    stores = layout.get("stores", layout)  # tolerate flat or nested shape
    if isinstance(stores, dict):
        return stores.get(store_id, {}) or {}
    if isinstance(stores, list):
        for s in stores:
            if s.get("store_id") == store_id:
                return s
    return {}


def is_open_at(store_id: str, when: dt.datetime) -> bool:
    """True if the store is open at ``when`` (UTC). Unknown layout => always open
    so we never suppress a genuine dead-zone signal."""
    block = _store_block(store_id)
    hours = block.get("open_hours")
    if not hours:
        return True
    try:
        open_h = _parse_hm(hours.get("open", "00:00"))
        close_h = _parse_hm(hours.get("close", "23:59"))
    except (ValueError, AttributeError):
        return True
    local = when.timetz()
    t = (local.hour, local.minute)
    return open_h <= t <= close_h


def zone_names(store_id: str) -> Optional[list[str]]:
    block = _store_block(store_id)
    zones = block.get("zones")
    if isinstance(zones, list):
        return [z if isinstance(z, str) else z.get("zone_id") for z in zones]
    if isinstance(zones, dict):
        return list(zones.keys())
    return None


def _parse_hm(s: str) -> tuple[int, int]:
    h, m = s.split(":")[:2]
    return int(h), int(m)
