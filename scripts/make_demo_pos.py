"""Generate an *illustrative* POS feed for the footage stores so the North Star
(offline conversion rate) is demonstrable end-to-end.

Why this exists: the hackathon-provided ``data/pos_transactions.csv`` is for store
``ST1008`` -- which has NO footage -- so it correlates to zero detected visitors and
conversion reads 0 even though the correlation mechanism (app/pos.py) is built and
unit-tested. To DEMONSTRATE that mechanism on the footage store(s), this script
derives a plausible POS feed from the billing-zone presences the pipeline actually
detected: a subset (~CONVERT_RATE) of non-staff visitors seen at billing "buy",
with a transaction a few seconds after their billing presence and a realistic basket
value. It is deterministic (seeded) so reruns are idempotent.

This is clearly an illustrative stand-in, not ground truth -- the brief states POS is
provided separately; here it is mapped onto the store we actually have video for.

    python scripts/make_demo_pos.py                 # writes data/pos_transactions.csv (legacy format)
    python scripts/make_demo_pos.py --convert 0.6   # tune the share of billing visitors who purchase

The provided line-item file is preserved as data/pos_provided_ST1008.csv (the
line-item loader is still exercised by tests/test_schema_adapter.py).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BILLING_TYPES = {"BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON"}


def _parse_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _billing_visitors(events: list[dict]) -> dict[str, dt.datetime]:
    """visitor_id -> earliest non-staff billing-zone timestamp, per store's events."""
    seen: dict[str, dt.datetime] = {}
    for e in events:
        if e.get("is_staff"):
            continue
        zone = (e.get("zone_id") or "").upper()
        if e.get("event_type") in BILLING_TYPES or "BILLING" in zone:
            ts = _parse_iso(e["timestamp"])
            cur = seen.get(e["visitor_id"])
            if cur is None or ts < cur:
                seen[e["visitor_id"]] = ts
    return seen


def _load_events(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def build_rows(events: list[dict], convert_rate: float, delay_s: int,
               seed: int) -> list[dict]:
    """One POS basket per *purchasing* billing visitor, grouped by store_id."""
    # group billing presences by store
    by_store: dict[str, dict[str, dt.datetime]] = {}
    for e in events:
        by_store.setdefault(e["store_id"], {})
    for store_id in by_store:
        store_events = [e for e in events if e["store_id"] == store_id]
        by_store[store_id] = _billing_visitors(store_events)

    rng = random.Random(seed)
    rows: list[dict] = []
    for store_id, visitors in sorted(by_store.items()):
        items = sorted(visitors.items(), key=lambda kv: kv[1])  # by time, deterministic
        n_buy = round(len(items) * convert_rate)
        # deterministically choose which billing visitors purchase
        buyers = sorted(rng.sample(items, n_buy), key=lambda kv: kv[1]) if n_buy else []
        for i, (vid, ts) in enumerate(buyers, 1):
            txn_ts = ts + dt.timedelta(seconds=delay_s)
            rows.append({
                "store_id": store_id,
                "transaction_id": f"{store_id}_TXN_{i:03d}",
                "timestamp": txn_ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                "basket_value_inr": round(rng.uniform(250, 2500), 2),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="*",
                    default=["data/events_store1.jsonl", "data/events_store2.jsonl"],
                    help="per-store events; falls back to data/events.jsonl if none exist")
    ap.add_argument("--out", default="data/pos_transactions.csv")
    ap.add_argument("--convert", type=float, default=0.65,
                    help="share of billing visitors who purchase (default 0.65)")
    ap.add_argument("--delay-s", type=int, default=45,
                    help="seconds between billing presence and the POS transaction "
                         "(must be < pos_correlation_window_min*60, default 45)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    events: list[dict] = []
    for p in a.events:
        events += _load_events(os.path.join(ROOT, p))
    if not events:
        events = _load_events(os.path.join(ROOT, "data/events.jsonl"))
    if not events:
        raise SystemExit("No events found; run the pipeline first.")

    rows = build_rows(events, a.convert, a.delay_s, a.seed)
    if not rows:
        raise SystemExit("No non-staff billing presences found -> no POS rows.")

    out = os.path.join(ROOT, a.out)
    # Preserve the hackathon-provided line-item POS the first time we overwrite it.
    provided = os.path.join(ROOT, "data/pos_provided_ST1008.csv")
    if os.path.exists(out) and not os.path.exists(provided):
        shutil.copyfile(out, provided)
        print(f"Preserved provided POS -> {provided}")

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["store_id", "transaction_id",
                                           "timestamp", "basket_value_inr"])
        w.writeheader()
        w.writerows(rows)

    per_store: dict[str, int] = {}
    for r in rows:
        per_store[r["store_id"]] = per_store.get(r["store_id"], 0) + 1
    print(f"Wrote {len(rows)} POS baskets -> {a.out}  "
          f"({', '.join(f'{k}:{v}' for k, v in sorted(per_store.items()))})")


if __name__ == "__main__":
    main()
