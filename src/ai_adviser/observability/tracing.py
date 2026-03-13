"""
OpenTelemetry tracing helpers for ai‑adviser.

This module provides a simple initialisation routine to instrument a FastAPI
application with OpenTelemetry and to create spans for internal RAG pipeline
operations.  Tracing is optional and controlled via environment variables
(see `settings.TRACING_ENABLED`).  When disabled all functions become
no‑ops.  When enabled the FastAPI instrumentation collects spans for each
incoming request and uses the OTLP exporter to send traces to the endpoint
configured in `settings.OTLP_ENDPOINT`.

To avoid import errors in environments without OpenTelemetry installed the
imports are wrapped in try/except blocks.  Missing dependencies result in
tracing being disabled.

This module defines:

* ``init_tracing(app)`` – call once at startup to configure tracing.
* ``get_tracer()`` – retrieve the tracer used for manual spans.

The `span` context manager lives in ``observability.langfuse`` and uses
``get_tracer()`` to create child spans for internal stages.  Manual spans
should be nested within the request span created by the instrumentation.
"""

from __future__ import annotations

import os
from typing import Optional

from ai_adviser.config import settings

try:
    # OpenTelemetry API and SDK modules
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover
    # When OpenTelemetry packages are unavailable, tracing remains disabled
    _OTEL_AVAILABLE = False


_tracer: Optional[trace.Tracer] = None


def init_tracing(app) -> None:
    """Initialise OpenTelemetry tracing for a FastAPI application.

    This function instruments FastAPI using the OpenTelemetry Instrumentor and
    configures an OTLP exporter pointing at the endpoint specified by
    ``settings.OTLP_ENDPOINT``.  When tracing is disabled via configuration or
    required modules are missing the function returns immediately.

    Args:
        app: The FastAPI application instance to instrument.
    """
    global _tracer
    if not settings.TRACING_ENABLED or not _OTEL_AVAILABLE:
        return

    # Ensure we only initialise tracing once
    if _tracer is not None:
        return

    # Configure resource attributes to identify the service
    resource = Resource.create({"service.name": "ai-adviser"})
    provider = TracerProvider(resource=resource)
    # Configure OTLP exporter if endpoint provided
    if settings.OTLP_ENDPOINT:
        exporter = OTLPSpanExporter(
            endpoint=settings.OTLP_ENDPOINT,
            timeout=10,
        )
        span_processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(span_processor)
    # Register as global provider
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(__name__)
    # Instrument the FastAPI app.  This middleware wraps every request in a span.
    try:
        FastAPIInstrumentor().instrument_app(app)
    except Exception:
        # Fallback to ASGI middleware if instrumentation fails
        app.add_middleware(OpenTelemetryMiddleware)


def get_tracer() -> Optional["trace.Tracer"]:
    """Return the tracer used for manual spans or None if tracing is disabled."""
    return _tracer