"""Step 9 + section 11's τ/k sweep: comparative experiments over a handful of
sample prompts — always-edge vs. the proposed threshold policy, and a sweep
of the policy's two parameters (threshold τ, window size k).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from .confidence import ConfidenceExtractor, ConfidenceMetric
from .middleware import GenerationResult, MiddlewareCore
from .model_backend import CloudModelAdapter, EdgeModelAdapter, ModelBackend
from .policy import DecisionPolicy

SAMPLE_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
    "In 1969, humans first landed on",
]

ALWAYS_EDGE_THRESHOLD = -1.0  # margin confidence is always >= 0 -> never switches

# Default sweep grid (section 7: k=4..8; a handful of thresholds spanning
# margin confidence's [0, 1] range).
DEFAULT_THRESHOLDS = (0.2, 0.5, 0.8)
DEFAULT_WINDOW_SIZES = (4, 6, 8)


def _timed_generate(
    edge: ModelBackend,
    cloud: ModelBackend,
    confidence: ConfidenceExtractor,
    prompt: str,
    threshold: float,
    window_size: int,
    max_new_tokens: int,
) -> tuple[float, GenerationResult]:
    middleware = MiddlewareCore(
        edge_backend=edge,
        cloud_backend=cloud,
        confidence_extractor=confidence,
        policy=DecisionPolicy(threshold=threshold, window_size=window_size),
        max_new_tokens=max_new_tokens,
    )
    start = time.perf_counter()
    result = middleware.generate(prompt)
    return time.perf_counter() - start, result


@dataclass
class ComparisonRow:
    prompt: str
    policy: str
    latency_seconds: float
    tokens_generated: int
    switch_count: int
    cloud_fallback_count: int
    text: str


def run_comparison(
    prompts: list[str] | None = None,
    threshold: float = 0.5,
    window_size: int = 4,
    max_new_tokens: int = 40,
) -> list[ComparisonRow]:
    prompts = prompts if prompts is not None else SAMPLE_PROMPTS

    edge = EdgeModelAdapter()
    cloud = CloudModelAdapter()
    confidence = ConfidenceExtractor(metric=ConfidenceMetric.MARGIN)

    rows: list[ComparisonRow] = []
    for prompt in prompts:
        for policy_name, policy_threshold in [
            ("always_edge", ALWAYS_EDGE_THRESHOLD),
            ("threshold_policy", threshold),
        ]:
            latency, result = _timed_generate(
                edge, cloud, confidence, prompt, policy_threshold, window_size, max_new_tokens
            )
            rows.append(
                ComparisonRow(
                    prompt=prompt,
                    policy=policy_name,
                    latency_seconds=latency,
                    tokens_generated=len(result.log.events()),
                    switch_count=result.switch_count,
                    cloud_fallback_count=result.cloud_fallback_count,
                    text=result.text,
                )
            )
    return rows


@dataclass
class SweepRow:
    prompt: str
    threshold: float
    window_size: int
    latency_seconds: float
    tokens_generated: int
    switch_count: int
    cloud_fallback_count: int
    text: str


def run_sweep(
    prompts: list[str] | None = None,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    window_sizes: tuple[int, ...] = DEFAULT_WINDOW_SIZES,
    max_new_tokens: int = 40,
    edge: ModelBackend | None = None,
    cloud: ModelBackend | None = None,
    confidence: ConfidenceExtractor | None = None,
) -> list[SweepRow]:
    """Runs every (prompt, τ, k) combination once and records latency,
    length, and switch count — the raw data for section 11's τ/k sweep."""
    prompts = prompts if prompts is not None else SAMPLE_PROMPTS
    edge = edge if edge is not None else EdgeModelAdapter()
    cloud = cloud if cloud is not None else CloudModelAdapter()
    confidence = confidence if confidence is not None else ConfidenceExtractor(metric=ConfidenceMetric.MARGIN)

    rows: list[SweepRow] = []
    for prompt in prompts:
        for threshold in thresholds:
            for window_size in window_sizes:
                latency, result = _timed_generate(
                    edge, cloud, confidence, prompt, threshold, window_size, max_new_tokens
                )
                rows.append(
                    SweepRow(
                        prompt=prompt,
                        threshold=threshold,
                        window_size=window_size,
                        latency_seconds=latency,
                        tokens_generated=len(result.log.events()),
                        switch_count=result.switch_count,
                        cloud_fallback_count=result.cloud_fallback_count,
                        text=result.text,
                    )
                )
    return rows


def rows_to_dicts(rows: list[ComparisonRow] | list[SweepRow]) -> list[dict]:
    return [asdict(row) for row in rows]
