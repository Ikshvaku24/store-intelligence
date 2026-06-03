"""FastAPI application: router wiring + middleware registration (BUILD_SPEC Section 4).

The graded, lightweight Process 2. No torch/opencv -- it only ingests events and
serves query-time aggregations.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import anomalies, db, funnel, health, heatmap, ingestion, metrics, pos
from app.config import get_settings
from app.logging_mw import StructuredLoggingMiddleware, register_exception_handlers

app = FastAPI(title="Store Intelligence API", version="1.0.0")
app.add_middleware(StructuredLoggingMiddleware)
# Allow the static dashboard (served on a different port) to poll the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
register_exception_handlers(app)


@app.on_event("startup")
def _startup() -> None:
    # Idempotent: create tables if absent, safe to run repeatedly.
    db.init_db()
    # Auto-load POS transactions if present (optional, never fatal).
    csv_path = os.environ.get("POS_CSV_PATH", "data/pos_transactions.csv")
    if os.path.exists(csv_path):
        try:
            pos.load_pos_csv(csv_path)
        except Exception:  # noqa: BLE001 - POS load is best-effort at startup
            pass


@app.get("/health")
def get_health() -> dict[str, Any]:
    return health.compute_health()


@app.post("/events/ingest")
async def post_ingest(request: Request) -> JSONResponse:
    trace_id = getattr(request.state, "trace_id", "")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_json", "detail": "Request body is not valid JSON.", "trace_id": trace_id},
        )

    if isinstance(body, dict) and "events" in body:
        raw_events = body["events"]
    elif isinstance(body, list):
        raw_events = body
    else:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_payload",
                "detail": "Expected a JSON array of events or {\"events\": [...]}.",
                "trace_id": trace_id,
            },
        )

    request.state.event_count = len(raw_events) if isinstance(raw_events, list) else 0
    result = ingestion.ingest_batch(raw_events if isinstance(raw_events, list) else [], trace_id)
    status = result.pop("_status", 200)
    return JSONResponse(status_code=status, content=result)


@app.get("/stores/{id}/metrics")
def get_metrics(id: str, window_min: Optional[int] = Query(default=None, ge=1)) -> dict[str, Any]:
    return metrics.compute_metrics(id, window_min)


@app.get("/stores/{id}/funnel")
def get_funnel(id: str, window_min: Optional[int] = Query(default=None, ge=1)) -> dict[str, Any]:
    return funnel.compute_funnel(id, window_min)


@app.get("/stores/{id}/heatmap")
def get_heatmap(id: str, window_min: Optional[int] = Query(default=None, ge=1)) -> dict[str, Any]:
    return heatmap.compute_heatmap(id, window_min)


@app.get("/stores/{id}/anomalies")
def get_anomalies(id: str, window_min: Optional[int] = Query(default=None, ge=1)) -> dict[str, Any]:
    return anomalies.compute_anomalies(id, window_min)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "store-intelligence", "version": app.version, "docs": "/docs"}
