"""Lightweight metrics collection for observability."""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional
from contextlib import contextmanager

from app.core.config import settings


@dataclass
class Counter:
    """Thread-safe counter metric."""
    value: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def increment(self, amount: int = 1) -> None:
        with self._lock:
            self.value += amount

    def get(self) -> int:
        with self._lock:
            return self.value


@dataclass
class Histogram:
    """Thread-safe histogram metric for latency measurements."""
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _values: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        with self._lock:
            self._values.append(value)

    def get_values(self) -> list[float]:
        with self._lock:
            return self._values.copy()

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


@dataclass
class Metrics:
    """Collection of application metrics."""

    # HTTP metrics
    http_requests_total: Counter = field(default_factory=Counter)
    http_request_duration_seconds: Histogram = field(default_factory=Histogram)
    http_requests_by_status: Dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))

    # RAG metrics
    rag_requests_total: Counter = field(default_factory=Counter)
    rag_request_duration_seconds: Histogram = field(default_factory=Histogram)
    rag_retrieval_latency_seconds: Histogram = field(default_factory=Histogram)
    rag_grading_latency_seconds: Histogram = field(default_factory=Histogram)
    rag_generation_latency_seconds: Histogram = field(default_factory=Histogram)
    rag_grounding_latency_seconds: Histogram = field(default_factory=Histogram)
    rag_retry_count_total: Counter = field(default_factory=Counter)
    rag_healing_count_total: Counter = field(default_factory=Counter)
    rag_failure_by_category: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    def _get_status_counter(self, status_code: int) -> Counter:
        """Get or create counter for HTTP status code."""
        if status_code not in self.http_requests_by_status:
            self.http_requests_by_status[status_code] = Counter()
        return self.http_requests_by_status[status_code]

    def _get_failure_counter(self, category: str) -> Counter:
        """Get or create counter for failure category."""
        if category not in self.rag_failure_by_category:
            self.rag_failure_by_category[category] = Counter()
        return self.rag_failure_by_category[category]


# Global metrics instance
_metrics: Optional[Metrics] = None
_metrics_lock = threading.Lock()


def get_metrics() -> Metrics:
    """Get the global metrics instance (thread-safe singleton)."""
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = Metrics()
    return _metrics


@contextmanager
def timer(histogram: Histogram):
    """Context manager to time operations and record to histogram."""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        histogram.observe(duration)


def increment_counter(counter: Counter, amount: int = 1) -> None:
    """Increment a counter by amount."""
    counter.increment(amount)


def observe_histogram(histogram: Histogram, value: float) -> None:
    """Observe a value in a histogram."""
    histogram.observe(value)