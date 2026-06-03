"""Database engine, schema, and a thin repository layer.

Design notes
------------
* SQLAlchemy Core (not the ORM) keeps the SQL explicit and swappable.
* The same code runs on **Postgres** (docker/production -- native conflict-ignore,
  the credible scaling story) and **SQLite** (local/test -- no server needed).
  Dialect-aware `insert(...).on_conflict_do_nothing()` is selected per engine.
* Timestamps are stored as **naive UTC** so range filters behave identically on
  both dialects; the API serialises them back with a trailing ``Z``.
* Query-time aggregation is done in Python over fetched rows. The dataset is
  tiny (one store, ~90s) so this is simpler and fully portable; the indexes
  below keep it fast at production volume.
"""
from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from typing import Any, Iterable, Optional, Sequence

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.types import JSON

from app.config import get_settings
from app.normalize import canonical_store_id

metadata_obj = MetaData()

events_table = Table(
    "events",
    metadata_obj,
    Column("event_id", String, primary_key=True),
    Column("store_id", String, nullable=False, index=True),
    Column("camera_id", String, nullable=False),
    Column("visitor_id", String, nullable=False),
    Column("event_type", String, nullable=False),
    Column("timestamp", DateTime, nullable=False),
    Column("zone_id", String, nullable=True),
    Column("dwell_ms", Integer, nullable=False, default=0),
    Column("is_staff", Boolean, nullable=False, default=False, index=True),
    Column("confidence", Float, nullable=False),
    Column("metadata", JSON, nullable=False, default=dict),
)

pos_table = Table(
    "pos_transactions",
    metadata_obj,
    Column("transaction_id", String, primary_key=True),
    Column("store_id", String, nullable=False, index=True),
    Column("timestamp", DateTime, nullable=False),
    Column("basket_value_inr", Numeric, nullable=False),
)


# Composite indexes (named so create_all emits them on both dialects).
from sqlalchemy import Index  # noqa: E402

Index("ix_events_store_ts", events_table.c.store_id, events_table.c.timestamp)
Index("ix_events_store_visitor", events_table.c.store_id, events_table.c.visitor_id)
Index("ix_events_store_type", events_table.c.store_id, events_table.c.event_type)
Index("ix_pos_store_ts", pos_table.c.store_id, pos_table.c.timestamp)


_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = get_settings().db_url
        connect_args: dict[str, Any] = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(url, future=True, pool_pre_ping=True, connect_args=connect_args)
    return _engine


def reset_engine() -> None:
    """Drop the cached engine (used by tests that swap DB_URL)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def init_db() -> None:
    """Create tables/indexes if absent. Safe to call repeatedly (idempotent)."""
    metadata_obj.create_all(get_engine())


@contextmanager
def connection():
    eng = get_engine()
    with eng.begin() as conn:
        yield conn


# --------------------------------------------------------------------------- #
# Dialect-aware bulk upsert
# --------------------------------------------------------------------------- #
def _insert_stmt(table: Table):
    name = get_engine().dialect.name
    if name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        return pg_insert(table)
    if name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        return sqlite_insert(table)
    # Generic fallback: caller must pre-filter duplicates.
    from sqlalchemy import insert as generic_insert

    return generic_insert(table)


def insert_events_ignore_conflicts(rows: Sequence[dict[str, Any]]) -> None:
    """Bulk-insert events; rows whose event_id already exists are ignored."""
    if not rows:
        return
    stmt = _insert_stmt(events_table)
    if hasattr(stmt, "on_conflict_do_nothing"):
        stmt = stmt.on_conflict_do_nothing(index_elements=["event_id"])
    with connection() as conn:
        conn.execute(stmt, list(rows))


def insert_pos_ignore_conflicts(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    stmt = _insert_stmt(pos_table)
    if hasattr(stmt, "on_conflict_do_nothing"):
        stmt = stmt.on_conflict_do_nothing(index_elements=["transaction_id"])
    with connection() as conn:
        conn.execute(stmt, list(rows))


def existing_event_ids(event_ids: Iterable[str]) -> set[str]:
    ids = list({e for e in event_ids})
    if not ids:
        return set()
    found: set[str] = set()
    with connection() as conn:
        # Chunk to stay clear of parameter limits on large batches.
        for i in range(0, len(ids), 900):
            chunk = ids[i : i + 900]
            rows = conn.execute(
                select(events_table.c.event_id).where(events_table.c.event_id.in_(chunk))
            ).all()
            found.update(r[0] for r in rows)
    return found


# --------------------------------------------------------------------------- #
# Read repository (used by metrics/funnel/heatmap/anomalies)
# --------------------------------------------------------------------------- #
def _row_to_dict(row) -> dict[str, Any]:
    d = dict(row._mapping)
    ts = d.get("timestamp")
    if isinstance(ts, dt.datetime) and ts.tzinfo is None:
        d["timestamp"] = ts.replace(tzinfo=dt.timezone.utc)
    if d.get("metadata") is None:
        d["metadata"] = {}
    return d


def fetch_events(
    store_id: str,
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None,
    event_types: Optional[Sequence[str]] = None,
    include_staff: bool = True,
) -> list[dict[str, Any]]:
    q = select(events_table).where(events_table.c.store_id == canonical_store_id(store_id))
    if start is not None:
        q = q.where(events_table.c.timestamp >= _naive(start))
    if end is not None:
        q = q.where(events_table.c.timestamp <= _naive(end))
    if event_types:
        q = q.where(events_table.c.event_type.in_(list(event_types)))
    if not include_staff:
        q = q.where(events_table.c.is_staff.is_(False))
    q = q.order_by(events_table.c.timestamp)
    with connection() as conn:
        return [_row_to_dict(r) for r in conn.execute(q).all()]


def fetch_pos(
    store_id: str,
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None,
) -> list[dict[str, Any]]:
    q = select(pos_table).where(pos_table.c.store_id == canonical_store_id(store_id))
    if start is not None:
        q = q.where(pos_table.c.timestamp >= _naive(start))
    if end is not None:
        q = q.where(pos_table.c.timestamp <= _naive(end))
    q = q.order_by(pos_table.c.timestamp)
    with connection() as conn:
        return [_row_to_dict(r) for r in conn.execute(q).all()]


def last_event_per_store() -> list[dict[str, Any]]:
    q = select(
        events_table.c.store_id,
        func.max(events_table.c.timestamp).label("last_event_ts"),
    ).group_by(events_table.c.store_id)
    with connection() as conn:
        out = []
        for r in conn.execute(q).all():
            ts = r[1]
            if isinstance(ts, dt.datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=dt.timezone.utc)
            out.append({"store_id": r[0], "last_event_ts": ts})
        return out


def db_ok() -> bool:
    try:
        with connection() as conn:
            conn.execute(select(1))
        return True
    except Exception:
        return False


def _naive(d: dt.datetime) -> dt.datetime:
    """Coerce to naive UTC to match stored timestamps."""
    if d.tzinfo is not None:
        d = d.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return d
