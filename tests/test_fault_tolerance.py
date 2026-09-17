"""run_with_timeout must actually return within ~timeout_seconds even when
the underlying call keeps running past the deadline (regression test for a
bug where `with ThreadPoolExecutor()` blocked on shutdown until the slow
call finished, making the timeout a no-op)."""

import time

import pytest

from edge_cloud_llm.fault_tolerance import (
    CloudUnavailableError,
    call_cloud_with_fallback,
    run_with_timeout,
)


def _slow_call():
    time.sleep(2.0)
    return "done"


def test_run_with_timeout_returns_promptly_on_timeout():
    start = time.perf_counter()
    with pytest.raises(CloudUnavailableError):
        run_with_timeout(_slow_call, timeout_seconds=0.1)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"timeout took {elapsed:.2f}s wall-clock, expected well under the 2s call itself"


def test_run_with_timeout_returns_result_when_within_budget():
    assert run_with_timeout(_slow_call, timeout_seconds=5.0) == "done"


def test_call_cloud_with_fallback_falls_back_promptly_on_timeout():
    start = time.perf_counter()
    result = call_cloud_with_fallback(_slow_call, fallback_call=lambda: "fallback", timeout_seconds=0.1)
    elapsed = time.perf_counter() - start
    assert result == "fallback"
    assert elapsed < 1.0
