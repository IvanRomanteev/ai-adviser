"""
Prometheus metrics helpers for ai‑adviser.

This module defines metrics collectors and provides helper functions for
recording counters and timings.  Metrics can be enabled or disabled via
environment variables (see `settings.METRICS_ENABLED`).  When disabled the
helpers fall back to in‑memory counters to allow unit tests to inspect
metrics without requiring a Prometheus client.  When enabled the
`prometheus_client` library is used to register counters and histograms
exposed via the `/metrics` endpoint.

To minimise coupling between the service code and the monitoring backend,
metrics should be recorded via the `record_metric` function and timings
should be recorded via the `record_timing` helper or automatically by
wrapping code in the `span` context manager (see `observability.langfuse`).

At application startup the `setup_metrics` function should be invoked to
attach the `/metrics` endpoint to the FastAPI application.  The route will
only be registered when metrics are enabled.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

from fastapi import APIRouter, Response

from ai_adviser.config import settings

try:
    # Import Prometheus client if available.  The agent does not mandate its
    # presence and will gracefully degrade to a stub if import fails.
    from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when prometheus_client is missing
    CollectorRegistry = None  # type: ignore
    Counter = None  # type: ignore
    Histogram = None  # type: ignore
    generate_latest = None  # type: ignore
    _PROM_AVAILABLE = False

# In‑memory fallback storage for simple metrics used in unit tests.  Even when
# Prometheus is enabled these counters are still updated to allow existing
# tests to inspect metric values via `get_metric`.
_metrics_lock = None  # type: Optional[object]
_metrics: Dict[str, float] = {}

try:
    import threading as _threading

    _metrics_lock = _threading.Lock()
except Exception:
    # Should never happen but fallback to dummy lock if threading is unavailable
    class DummyLock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    _metrics_lock = DummyLock()


class MetricsCollector:
    """Container for Prometheus collectors used by ai‑adviser.

    When metrics are enabled this class registers counters and histograms on
    module import.  Labels are chosen conservatively to avoid cardinality
    explosion.  All buckets are expressed in seconds.
    """

    def __init__(self) -> None:
        self.enabled = settings.METRICS_ENABLED and _PROM_AVAILABLE
        if not self.enabled:
            # Register nothing if disabled
            self.registry: Optional[CollectorRegistry] = None
            self.request_latency: Optional[Histogram] = None
            self.stage_latency: Optional[Histogram] = None
            self.counters: Dict[str, Counter] = {}
            return

        # Create a registry isolated from default to avoid double collecting
        self.registry = CollectorRegistry(auto_describe=True)
        # HTTP request latency histogram.  Use a few buckets to estimate p50/p95/p99.
        self.request_latency = Histogram(
            name="http_request_latency_seconds",
            documentation="HTTP request latency by endpoint, method and status code",
            labelnames=("endpoint", "method", "status_code"),
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )
        # Stage latency histogram: tracks durations of internal RAG pipeline stages
        self.stage_latency = Histogram(
            name="rag_stage_latency_seconds",
            documentation="Duration of RAG pipeline stages",
            labelnames=("stage",),
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )
        # Generic counters for calls, errors, fallbacks and retries
        self.counters: Dict[str, Counter] = {}
        for name, doc in [
            ("rag_chat_calls", "Number of calls to the /rag_chat endpoint"),
            ("embed_calls", "Number of embedding calls"),
            ("search_calls", "Number of search calls"),
            ("generate_calls", "Number of generation calls"),
            ("error_count", "Number of exceptions raised during request handling"),
            ("fallback_no_hits", "Number of times the RAG pipeline returned no hits"),
            ("fallback_citation_fail", "Number of times fallback occurred due to citation validation failure"),
            ("fallback_citation_structure_fail", "Number of times fallback occurred due to citation structure validation failure"),
            ("fallback_irrelevant_hits", "Number of times fallback occurred because retrieved hits were irrelevant"),
            ("retry_count", "Number of retry attempts across external calls"),
        ]:
            self.counters[name] = Counter(
                name=name,
                documentation=doc,
                registry=self.registry,
            )

    def increment(self, name: str, value: float = 1.0) -> None:
        # Update Prometheus counter if present
        if self.enabled:
            counter = self.counters.get(name)
            if counter is not None:
                try:
                    counter.inc(value)
                except Exception:
                    pass
        # Always update in‑memory counter for tests
        with _metrics_lock:
            _metrics[name] = _metrics.get(name, 0.0) + value

    def observe_stage(self, stage: str, seconds: float) -> None:
        if self.enabled and self.stage_latency is not None:
            try:
                self.stage_latency.labels(stage).observe(seconds)
            except Exception:
                pass
        # Also record aggregated stage durations under a per‑stage metric for tests
        with _metrics_lock:
            key = f"{stage}_seconds_total"
            _metrics[key] = _metrics.get(key, 0.0) + seconds

    def observe_request(self, endpoint: str, method: str, status_code: int, seconds: float) -> None:
        if self.enabled and self.request_latency is not None:
            try:
                self.request_latency.labels(endpoint, method, str(status_code)).observe(seconds)
            except Exception:
                pass
        # update total request latency per endpoint for tests
        with _metrics_lock:
            key = f"{endpoint}:{method}:{status_code}:seconds_total"
            _metrics[key] = _metrics.get(key, 0.0) + seconds

    def as_router(self) -> Optional[APIRouter]:
        """Return a router exposing the /metrics endpoint or None.

        When metrics are disabled this returns None so that no route is added.
        """
        if not self.enabled or not self.registry or not generate_latest:
            return None

        router = APIRouter()

        @router.get("/metrics")
        def metrics_endpoint() -> Response:  # pragma: no cover - tested via functional tests
            # Expose the metrics as plain text
            data = generate_latest(self.registry)
            return Response(content=data, media_type="text/plain; charset=utf-8")

        return router


# Singleton collector used throughout the application
collector = MetricsCollector()


def setup_metrics(app) -> None:
    """Attach the /metrics route to the FastAPI application.

    Should be called once at startup.  When metrics are disabled no route is
    added.
    """
    router = collector.as_router()
    if router is not None:
        # Include metrics as a sub‑application to avoid prefixing paths
        app.include_router(router)


def record_metric(name: str, value: float) -> None:
    """Compatibility helper to increment a named counter.

    This wraps the MetricsCollector singleton.  Existing code uses this
    function to increment counters such as `rag_chat_calls`.  Calls to this
    function are forwarded to the collector and also update the in‑memory
    dictionary used in unit tests.
    """
    collector.increment(name, value)


def get_metric(name: str) -> float:
    """Return the current value of an in‑memory metric.

    Primarily intended for use in unit tests.  When metrics are enabled the
    Prometheus counters and histograms are authoritative; however this helper
    still reflects the aggregated total values recorded via the fallback
    dictionary.
    """
    with _metrics_lock:
        return _metrics.get(name, 0.0)


def reset_metrics() -> None:
    """Reset all in‑memory metric counters.

    This does not clear the Prometheus registry and is intended solely for
    unit tests where global state must be reset between tests.
    """
    with _metrics_lock:
        _metrics.clear()