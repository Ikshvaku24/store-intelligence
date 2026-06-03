"""POS load + time-window conversion correlation (BUILD_SPEC Section 10.7).

There is no ``customer_id`` in the POS feed, so conversion is established by
**store + time window**: a visitor present in the billing zone within the
configured window *before* a transaction's timestamp is counted as converted.
This keeps conversion robust on a tiny dataset without solving identity.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
from typing import Any, Iterable, Optional

from app import db
from app.config import get_settings
from app.normalize import canonical_store_id

BILLING_EVENT_TYPES = {"BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON"}


def load_pos_csv(path: str) -> int:
    """Load a POS CSV into the DB (idempotent). Returns transactions stored.

    Supports two formats, auto-detected from the header:

    * **legacy** -- one row per basket:
      ``store_id, transaction_id, timestamp, basket_value_inr``.
    * **hackathon** -- one row per *line item*:
      ``order_id, order_date(DD-MM-YYYY), order_time(HH:MM:SS), store_id,
      product_id, brand_name, total_amount``. Line items sharing a
      ``(store_id, order_date, order_time)`` are one basket; we collapse them into
      a single transaction whose value is the summed ``total_amount`` and whose id
      is deterministic (so re-loading is idempotent).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        if "order_id" in fields or "order_time" in fields:
            rows = _load_lineitem_format(reader)
        else:
            rows = _load_legacy_format(reader)
    db.insert_pos_ignore_conflicts(rows)
    return len(rows)


def _load_legacy_format(reader: "csv.DictReader") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in reader:
        ts = _parse_ts(r["timestamp"])
        rows.append({
            "transaction_id": r["transaction_id"].strip(),
            "store_id": canonical_store_id(r["store_id"].strip()),
            "timestamp": ts.replace(tzinfo=None),
            "basket_value_inr": float(r.get("basket_value_inr") or 0),
        })
    return rows


def _load_lineitem_format(reader: "csv.DictReader") -> list[dict[str, Any]]:
    # Aggregate line items into baskets keyed by (store, date, time).
    baskets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in reader:
        store = canonical_store_id((r.get("store_id") or "").strip())
        date_s = (r.get("order_date") or "").strip()
        time_s = (r.get("order_time") or "").strip()
        key = (store, date_s, time_s)
        b = baskets.get(key)
        if b is None:
            b = {
                "transaction_id": f"{store}_{date_s}_{time_s}".replace(" ", ""),
                "store_id": store,
                "timestamp": _parse_local(date_s, time_s),
                "basket_value_inr": 0.0,
            }
            baskets[key] = b
        try:
            b["basket_value_inr"] += float(r.get("total_amount") or 0)
        except ValueError:
            pass
    return list(baskets.values())


def _billing_presence(events: Iterable[dict[str, Any]]) -> list[tuple[str, dt.datetime]]:
    """(visitor_id, timestamp) for non-staff billing-zone presence."""
    out = []
    for e in events:
        if e.get("is_staff"):
            continue
        zone = (e.get("zone_id") or "").upper()
        if e["event_type"] in BILLING_EVENT_TYPES or "BILLING" in zone:
            out.append((e["visitor_id"], e["timestamp"]))
    return out


def correlate_conversions(
    store_id: str,
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None,
) -> dict[str, Any]:
    """Return converted visitor set + purchase count for the window."""
    settings = get_settings()
    window = dt.timedelta(minutes=settings.pos_correlation_window_min)

    # A purchase completes *after* the customer's last detected billing event, so
    # extend the POS fetch by the correlation window past the event window's end;
    # presence is still required to fall within [start, end].
    pos_end = end + window if end is not None else None
    txns = db.fetch_pos(store_id, start, pos_end)
    events = db.fetch_events(store_id, start, end, include_staff=False)
    presence = _billing_presence(events)

    converted: set[str] = set()
    for txn in txns:
        t_ts = txn["timestamp"]
        lo = t_ts - window
        for visitor_id, p_ts in presence:
            if lo <= p_ts <= t_ts:
                converted.add(visitor_id)

    return {
        "converted_visitors": converted,
        "purchase_count": len(txns),
        "basket_total_inr": float(sum(float(t["basket_value_inr"]) for t in txns)),
    }


def _parse_ts(s: str) -> dt.datetime:
    s = s.strip().replace("Z", "+00:00")
    d = dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def _parse_local(date_s: str, time_s: str) -> dt.datetime:
    """Parse a ``DD-MM-YYYY`` date + ``HH:MM:SS`` time into a naive datetime.

    Kept naive (store-local wall clock) to match the event timestamps it is
    correlated against -- conversion is a *relative* window comparison, so both
    sides only need to share one clock, not a particular timezone.
    """
    for fmt in ("%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(f"{date_s} {time_s}".strip(), fmt)
        except ValueError:
            continue
    # Last resort: ISO-ish fallback so a bad row doesn't crash the whole load.
    return dt.datetime.fromisoformat(f"{date_s}T{time_s}")
