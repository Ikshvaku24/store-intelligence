"""Stream events.jsonl -> POST /events/ingest at real or accelerated speed
(BUILD_SPEC Section 4 / Part E).

This is the "simulated real-time" driver: the same events.jsonl artifact that
powers batch scoring is replayed into the live API so the dashboard's counters
move. Inter-event gaps are preserved and divided by --speed.

    python dashboard/replay.py --events data/sample_events.jsonl --api http://localhost:8000 --speed 20
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from typing import Any

import httpx


def _parse_ts(s: str) -> dt.datetime:
    s = s.replace("Z", "+00:00")
    d = dt.datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def load_events(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8-sig") as fh:
        events = [json.loads(line) for line in fh if line.strip()]
    events.sort(key=lambda e: e["timestamp"])
    return events


def replay(events: list[dict[str, Any]], api: str, speed: float, batch: int) -> None:
    if not events:
        print("No events to replay.")
        return

    url = api.rstrip("/") + "/events/ingest"
    t_prev = _parse_ts(events[0]["timestamp"])
    pending: list[dict[str, Any]] = []
    sent = 0

    with httpx.Client(timeout=10.0) as client:
        for ev in events:
            t_cur = _parse_ts(ev["timestamp"])
            gap = (t_cur - t_prev).total_seconds() / max(speed, 0.001)
            if gap > 0 and pending:
                _flush(client, url, pending)
                sent += len(pending)
                pending = []
                time.sleep(min(gap, 2.0))  # cap any single sleep for a snappy demo
            pending.append(ev)
            t_prev = t_cur
            if len(pending) >= batch:
                _flush(client, url, pending)
                sent += len(pending)
                pending = []

        if pending:
            _flush(client, url, pending)
            sent += len(pending)

    print(f"Replayed {sent} events into {url} (speed x{speed}).")


def _flush(client: httpx.Client, url: str, batch: list[dict[str, Any]]) -> None:
    try:
        r = client.post(url, json=batch)
        if r.status_code >= 400:
            print(f"  ingest error {r.status_code}: {r.text[:160]}")
        else:
            body = r.json()
            print(f"  +{body.get('accepted', 0)} accepted, {body.get('duplicates', 0)} dup")
    except httpx.HTTPError as exc:
        print(f"  request failed: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/events.jsonl")
    ap.add_argument("--fallback", default="data/sample_events.jsonl",
                    help="used if --events does not exist")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--speed", type=float, default=20.0)
    ap.add_argument("--batch", type=int, default=25)
    args = ap.parse_args()

    import os

    path = args.events if os.path.exists(args.events) else args.fallback
    print(f"Replaying {path} ...")
    replay(load_events(path), args.api, args.speed, args.batch)


if __name__ == "__main__":
    main()
