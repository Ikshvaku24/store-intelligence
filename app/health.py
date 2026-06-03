"""Health endpoint (BUILD_SPEC Section 10.6).

Reports true DB connectivity and real last-event lag per store. This is what an
on-call engineer checks first, so it must be accurate -- stale_feed is computed
from the actual newest event timestamp, not a heartbeat.
"""
from __future__ import annotations

from app import db
from app.config import get_settings
from app.metrics import now_utc


def compute_health() -> dict:
    settings = get_settings()
    db_up = db.db_ok()
    stores = []
    if db_up:
        now = now_utc()
        for row in db.last_event_per_store():
            last_ts = row["last_event_ts"]
            lag = (now - last_ts).total_seconds()
            stores.append(
                {
                    "store_id": row["store_id"],
                    "last_event_ts": last_ts.isoformat().replace("+00:00", "Z"),
                    "lag_seconds": round(lag, 1),
                    "stale_feed": lag > settings.stale_feed_min * 60,
                }
            )

    from app import __version__

    return {
        "status": "ok" if db_up else "degraded",
        "db": "ok" if db_up else "down",
        "stores": stores,
        "version": __version__,
    }
