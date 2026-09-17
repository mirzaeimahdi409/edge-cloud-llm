"""Fault Tolerance Module: cloud call fails/times out -> fall back to edge.

Section 5 of CLAUDE.md is explicit: try/except + timeout is enough here, no
circuit breaker or separate infrastructure.
"""

from __future__ import annotations

import concurrent.futures
from typing import Callable, TypeVar

T = TypeVar("T")

DEFAULT_CLOUD_TIMEOUT_SECONDS = 5.0


class CloudUnavailableError(Exception):
    """Raised by run_with_timeout when the cloud call fails or times out."""


def run_with_timeout(
    call: Callable[[], T], timeout_seconds: float = DEFAULT_CLOUD_TIMEOUT_SECONDS
) -> T:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(call)
        try:
            return future.result(timeout=timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - any failure means "cloud unavailable"
            raise CloudUnavailableError(str(exc)) from exc


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
