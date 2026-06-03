-- Reference DDL for the Postgres deployment (BUILD_SPEC Section 12).
-- The application also creates these via SQLAlchemy metadata.create_all() on
-- startup (idempotent), so this file is documentation + a manual-bootstrap path.

CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT PRIMARY KEY,                 -- idempotency key
    store_id    TEXT NOT NULL,
    camera_id   TEXT NOT NULL,
    visitor_id  TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    timestamp   TIMESTAMP NOT NULL,               -- naive UTC
    zone_id     TEXT,
    dwell_ms    INTEGER NOT NULL DEFAULT 0,
    is_staff    BOOLEAN NOT NULL DEFAULT FALSE,
    confidence  REAL NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_events_store_ts      ON events (store_id, timestamp);
CREATE INDEX IF NOT EXISTS ix_events_store_visitor ON events (store_id, visitor_id);
CREATE INDEX IF NOT EXISTS ix_events_store_type    ON events (store_id, event_type);
CREATE INDEX IF NOT EXISTS ix_events_is_staff      ON events (is_staff);

CREATE TABLE IF NOT EXISTS pos_transactions (
    transaction_id   TEXT PRIMARY KEY,
    store_id         TEXT NOT NULL,
    timestamp        TIMESTAMP NOT NULL,
    basket_value_inr NUMERIC NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_pos_store_ts ON pos_transactions (store_id, timestamp);
