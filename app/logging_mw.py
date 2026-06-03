"""Cross-cutting production concerns (BUILD_SPEC Section 11).

* One structured JSON log line per request (trace_id, store_id, endpoint,
  latency_ms, event_count, status_code).
* trace_id read from an inbound header if present, else generated; echoed back.
* DB-unavailable -> HTTP 503 with a structured body (no raw stack traces).
* A catch-all handler guarantees no stack trace ever reaches a client.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError, OperationalError
from starlette.middleware.base import BaseHTTPMiddleware

TRACE_HEADER = "X-Trace-Id"

logger = logging.getLogger("store_intelligence")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _log(payload: dict) -> None:
    logger.info(json.dumps(payload, default=str))


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        trace_id = request.headers.get(TRACE_HEADER) or uuid.uuid4().hex
        request.state.trace_id = trace_id
        request.state.event_count = None
        request.state.store_id = request.path_params.get("id") if request.path_params else None

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except (OperationalError, DBAPIError):
            status_code = 503
            response = JSONResponse(
                status_code=503,
                content={
                    "error": "database_unavailable",
                    "detail": "The datastore is temporarily unavailable.",
                    "trace_id": trace_id,
                },
            )
        except Exception:  # noqa: BLE001 - never leak a stack trace to a client
            status_code = 500
            logger.exception("unhandled_error trace_id=%s", trace_id)
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "internal_error",
                    "detail": "An unexpected error occurred.",
                    "trace_id": trace_id,
                },
            )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers[TRACE_HEADER] = trace_id
        _log(
            {
                "trace_id": trace_id,
                "endpoint": request.url.path,
                "method": request.method,
                "store_id": getattr(request.state, "store_id", None),
                "event_count": getattr(request.state, "event_count", None),
                "latency_ms": latency_ms,
                "status_code": status_code,
            }
        )
        return response


def register_exception_handlers(app: FastAPI) -> None:
    """Belt-and-braces handlers (the middleware covers most paths, but explicit
    handlers cover errors raised inside routing/dependency resolution)."""

    @app.exception_handler(OperationalError)
    @app.exception_handler(DBAPIError)
    async def _db_down(request: Request, exc: Exception):  # noqa: ANN001
        trace_id = getattr(request.state, "trace_id", uuid.uuid4().hex)
        return JSONResponse(
            status_code=503,
            content={
                "error": "database_unavailable",
                "detail": "The datastore is temporarily unavailable.",
                "trace_id": trace_id,
            },
            headers={TRACE_HEADER: trace_id},
        )
