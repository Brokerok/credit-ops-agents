"""Tracing with Pydantic Logfire (OpenTelemetry under the hood).

Enabled only when LOGFIRE_TOKEN is set, so local runs and CI stay dependency-light. With a token every run
becomes one trace: HTTP request -> graph nodes -> agent runs with prompts, tool calls, token usage and cost
-> SQL statements. Any OTel backend (Langfuse, Grafana Tempo, Honeycomb) works the same way through
`additional_span_processors`; Logfire is used here because it instruments Pydantic AI natively.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.config import settings
from app.logging import get_logger

log = get_logger("observability")


def setup_tracing(app: FastAPI) -> bool:
    if not settings.logfire_token:
        return False
    import logfire

    logfire.configure(
        token=settings.logfire_token,
        service_name="credit-ops-agents",
        service_version=app.version,
        environment=settings.environment,
        console=False,
    )
    logfire.instrument_pydantic_ai()  # agent runs, model requests, tool calls, usage
    logfire.instrument_fastapi(app, capture_headers=False)  # one span per request, PII stays out of headers
    logfire.instrument_sqlalchemy()  # every engine created afterwards, i.e. the one from app.db.get_engine
    log.info("tracing_enabled", backend="logfire", environment=settings.environment)
    return True
