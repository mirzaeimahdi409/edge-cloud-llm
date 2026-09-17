"""Fault Tolerance Module: cloud call fails/times out -> fall back to edge.

Section 5 of CLAUDE.md is explicit: try/except + timeout is enough here, no
circuit breaker or separate infrastructure.
"""

from __future__ import annotations

import concurrent.futures
from typing import Callable, TypeVar

T = TypeVar("T")

DEFAULT_CLOUD_TIMEOUT_SECONDS = 20.0


class CloudUnavailableError(Exception):
    """Raised by run_with_timeout when the cloud call fails or times out."""


def run_with_timeout(
    call: Callable[[], T], timeout_seconds: float = DEFAULT_CLOUD_TIMEOUT_SECONDS
) -> T:
    # Deliberately not `with ThreadPoolExecutor() as executor:` — the context
    # manager's __exit__ calls shutdown(wait=True), which blocks until the
    # submitted call actually finishes even after future.result() has timed
    # out, silently turning the "timeout" into a no-op. shutdown(wait=False)
    # lets the (still-running) worker thread finish on its own and be
    # garbage-collected once done; its result is simply discarded.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(call)
    try:
        result = future.result(timeout=timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - any failure means "cloud unavailable"
        executor.shutdown(wait=False)
        raise CloudUnavailableError(str(exc)) from exc
    executor.shutdown(wait=False)
    return result


def call_cloud_with_fallback(
    cloud_call: Callable[[], T],
    fallback_call: Callable[[], T],
    timeout_seconds: float = DEFAULT_CLOUD_TIMEOUT_SECONDS,
) -> T:
    """Run cloud_call; if it fails or exceeds timeout_seconds, run
    fallback_call (continue on the edge) instead."""
    try:
        return run_with_timeout(cloud_call, timeout_seconds)
    except CloudUnavailableError:
        return fallback_call()
