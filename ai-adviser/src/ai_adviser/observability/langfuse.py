from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from ai_adviser.observability.metrics import collector
from ai_adviser.observability.tracing import get_tracer


@contextmanager
def span(stage: str) -> Iterator[None]:
    start = time.perf_counter()
    tracer = get_tracer()

    if tracer is not None:
        with tracer.start_as_current_span(stage):
            try:
                yield
            finally:
                collector.observe_stage(stage, time.perf_counter() - start)
    else:
        try:
            yield
        finally:
            collector.observe_stage(stage, time.perf_counter() - start)
