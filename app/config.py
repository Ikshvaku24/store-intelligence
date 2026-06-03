"""Environment-driven settings (windows, thresholds, DB URL).

All tunables live here so reviewers can see -- and override via env -- every
magic number. Defaults are chosen for the *clip-scale* dataset (a ~90s window),
with production values noted in comments and in docs/DESIGN.md.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database -----------------------------------------------------------
    # Postgres in docker/production; a sqlite URL is honoured for local/test runs
    # so the suite is runnable without a database server.
    db_url: str = "postgresql+psycopg://store:store@localhost:5432/store_intelligence"

    # --- API limits ---------------------------------------------------------
    max_batch_size: int = 500

    # --- Conversion / POS correlation --------------------------------------
    # A visitor counts as converted if seen in the billing zone within this many
    # minutes BEFORE a POS transaction timestamp (no customer identity needed).
    pos_correlation_window_min: int = 5

    # --- Anomaly thresholds -------------------------------------------------
    # Queue depth at/above this is a BILLING_QUEUE_SPIKE.
    queue_spike_threshold: int = 4
    # No visits for this many minutes (during open hours) = DEAD_ZONE.
    # Default tuned for the short clips; production uses ~30 min (see DESIGN.md).
    dead_zone_window_min: int = 5
    # A conversion drop of this fraction vs baseline is flagged.
    conversion_drop_pct: float = 0.30
    # Minimum minutes of history before CONVERSION_DROP is meaningful.
    conversion_min_history_min: int = 30

    # --- Health -------------------------------------------------------------
    # Feed considered stale when last event lag exceeds this many minutes.
    stale_feed_min: int = 10

    # --- Heatmap ------------------------------------------------------------
    # Below this many sessions the heatmap is flagged data_confidence=LOW.
    heatmap_min_sessions: int = 20

    # --- Metrics window -----------------------------------------------------
    # Default lookback for /metrics when caller passes no window (minutes).
    default_metrics_window_min: int = 60

    # --- Dwell / queue tuning ----------------------------------------------
    dwell_threshold_ms: int = 30_000  # ZONE_DWELL cadence used by the pipeline

    # --- Store layout (open hours, zone names) -----------------------------
    store_layout_path: str = "data/store_layout.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
