"""Generate synthetic-but-realistic sample data for local dev, tests, and the
dashboard demo.

The challenge's held-out grading data (sample_events.jsonl, pos_transactions.csv,
store_layout.json) is not shipped in this repo. This script produces faithful
stand-ins that exercise every event type and metric path, keyed off the real
single-store reality (STORE_BLR_002, five camera roles). Re-running is
deterministic (seeded).

    python scripts/generate_sample_data.py

Writes: data/sample_events.jsonl, data/pos_transactions.csv, data/store_layout.json
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import random
import uuid

random.seed(42)

STORE = "STORE_BLR_002"
BASE = dt.datetime(2026, 4, 16, 8, 0, 0, tzinfo=dt.timezone.utc)

CAMS = {
    "ENTRY": "CAM_ENTRY_03",
    "FLOOR_SKINCARE": "CAM_SKIN_01",
    "FLOOR_MAKEUP": "CAM_MAKEUP_02",
    "BILLING": "CAM_BILL_05",
    "BACKROOM": "CAM_BACK_04",
}

SKIN_ZONES = ["SKINCARE_MOISTURISER", "SKINCARE_CLEANSER", "FRAGRANCE_TABLE"]
MAKEUP_ZONES = ["MAKEUP_LIPS", "MAKEUP_FOUNDATION"]


def iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def ev(t, etype, visitor, cam, zone=None, dwell=0, is_staff=False, conf=0.8, meta=None):
    m = {"session_seq": meta.pop("session_seq", None)} if meta else {}
    if meta:
        m.update(meta)
    m = {k: v for k, v in m.items() if v is not None}
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": STORE,
        "camera_id": cam,
        "visitor_id": visitor,
        "event_type": etype,
        "timestamp": iso(t),
        "zone_id": zone,
        "dwell_ms": dwell,
        "is_staff": is_staff,
        "confidence": round(conf, 2),
        "metadata": m,
    }


def vid() -> str:
    return f"VIS_{uuid.uuid4().hex[:8]}"


def build_events():
    events = []
    t0 = BASE

    # --- Two staff members, present in backroom (staff oracle) ---
    for s in range(2):
        staff = f"VIS_STAFF_{s}"
        events.append(ev(t0, "ENTRY", staff, CAMS["BACKROOM"], is_staff=True, conf=0.95))
        events.append(
            ev(
                t0 + dt.timedelta(seconds=20),
                "ZONE_DWELL",
                staff,
                CAMS["BACKROOM"],
                zone="BACKROOM",
                dwell=60000,
                is_staff=True,
                conf=0.95,
            )
        )

    # --- 12 genuine customers ---
    n_customers = 12
    purchasers = set()
    for i in range(n_customers):
        v = vid()
        enter_t = t0 + dt.timedelta(seconds=5 + i * 4)
        seq = 1
        idconf = round(random.uniform(0.55, 0.95), 2)
        events.append(
            ev(enter_t, "ENTRY", v, CAMS["ENTRY"], conf=round(random.uniform(0.6, 0.92), 2),
               meta={"session_seq": seq, "id_source": "within_camera", "id_confidence": idconf})
        )
        seq += 1
        cur = enter_t

        # Visit a floor zone
        if i % 2 == 0:
            cam, zones = CAMS["FLOOR_SKINCARE"], SKIN_ZONES
        else:
            cam, zones = CAMS["FLOOR_MAKEUP"], MAKEUP_ZONES
        zone = random.choice(zones)
        cur += dt.timedelta(seconds=random.randint(3, 8))
        events.append(ev(cur, "ZONE_ENTER", v, cam, zone=zone, conf=0.8, meta={"session_seq": seq, "sku_zone": zone}))
        seq += 1
        dwell = random.randint(20000, 75000)
        if dwell >= 30000:
            events.append(ev(cur + dt.timedelta(seconds=30), "ZONE_DWELL", v, cam, zone=zone, dwell=30000, conf=0.8, meta={"session_seq": seq, "sku_zone": zone}))
            seq += 1
        events.append(ev(cur + dt.timedelta(milliseconds=dwell), "ZONE_EXIT", v, cam, zone=zone, dwell=dwell, conf=0.78, meta={"session_seq": seq, "sku_zone": zone}))
        seq += 1
        cur += dt.timedelta(milliseconds=dwell)

        # ~60% head to billing
        goes_to_bill = random.random() < 0.6
        if goes_to_bill:
            cur += dt.timedelta(seconds=random.randint(2, 6))
            depth = random.randint(1, 6)
            events.append(ev(cur, "BILLING_QUEUE_JOIN", v, CAMS["BILLING"], zone="BILLING_QUEUE", conf=0.7, meta={"session_seq": seq, "queue_depth": depth}))
            seq += 1
            # ~75% of queue joiners purchase; rest abandon
            if random.random() < 0.75:
                purchasers.add((v, cur + dt.timedelta(seconds=random.randint(10, 40))))
            else:
                events.append(ev(cur + dt.timedelta(seconds=random.randint(8, 20)), "BILLING_QUEUE_ABANDON", v, CAMS["BILLING"], zone="BILLING_QUEUE", conf=0.6, meta={"session_seq": seq}))
                seq += 1

        # Exit
        exit_t = cur + dt.timedelta(seconds=random.randint(15, 45))
        events.append(ev(exit_t, "EXIT", v, CAMS["ENTRY"], conf=0.75, meta={"session_seq": seq}))

        # One customer re-enters (same visitor_id) -> REENTRY, not a new visitor
        if i == 3:
            re_t = exit_t + dt.timedelta(seconds=30)
            events.append(ev(re_t, "REENTRY", v, CAMS["ENTRY"], conf=0.62, meta={"id_source": "reentry_match", "id_confidence": 0.66}))
            events.append(ev(re_t + dt.timedelta(seconds=20), "EXIT", v, CAMS["ENTRY"], conf=0.7))

    events.sort(key=lambda e: e["timestamp"])
    return events, purchasers


def build_pos(purchasers):
    rows = []
    for v, ts in sorted(purchasers, key=lambda x: x[1]):
        rows.append(
            {
                "store_id": STORE,
                "transaction_id": f"TXN_{uuid.uuid4().hex[:10]}",
                "timestamp": iso(ts),
                "basket_value_inr": random.choice([499, 899, 1299, 1599, 2499, 3199]),
            }
        )
    return rows


def build_layout():
    return {
        "stores": {
            STORE: {
                "store_id": STORE,
                "open_hours": {"open": "00:00", "close": "23:59"},
                "zones": SKIN_ZONES + MAKEUP_ZONES + ["BILLING_QUEUE", "BACKROOM"],
                "cameras": CAMS,
            }
        }
    }


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)

    events, purchasers = build_events()
    with open(os.path.join(data_dir, "sample_events.jsonl"), "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")

    pos_rows = build_pos(purchasers)
    with open(os.path.join(data_dir, "pos_transactions.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["store_id", "transaction_id", "timestamp", "basket_value_inr"])
        w.writeheader()
        w.writerows(pos_rows)

    with open(os.path.join(data_dir, "store_layout.json"), "w", encoding="utf-8") as fh:
        json.dump(build_layout(), fh, indent=2)

    print(f"Wrote {len(events)} events, {len(pos_rows)} POS transactions, store_layout.json")


if __name__ == "__main__":
    main()
