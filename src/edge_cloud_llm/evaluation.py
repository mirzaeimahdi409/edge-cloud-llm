"""Step 9: small comparative experiment — always-edge vs. the proposed
threshold policy — over a handful of sample prompts (section 11's minimal
evaluation, before any baseline beyond always-edge/always-cloud is added).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from .confidence import ConfidenceExtractor, ConfidenceMetric
from .middleware import MiddlewareCore
from .model_backend import CloudModelAdapter, EdgeModelAdapter
from .policy import DecisionPolicy

SAMPLE_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
    "In 1969, humans first landed on",
]

ALWAYS_EDGE_THRESHOLD = -1.0  # margin confidence is always >= 0 -> never switches


@dataclass
class ComparisonRow:
    prompt: str
    policy: str
    latency_seconds: float
    tokens_generated: int
    switch_count: int
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
            middleware = MiddlewareCore(
                edge_backend=edge,
                cloud_backend=cloud,
                confidence_extractor=confidence,
                policy=DecisionPolicy(threshold=policy_threshold, window_size=window_size),
                max_new_tokens=max_new_tokens,
            )
            start = time.perf_counter()
            result = middleware.generate(prompt)
            latency = time.perf_counter() - start

            rows.append(
                ComparisonRow(
                    prompt=prompt,
                    policy=policy_name,
                    latency_seconds=latency,
                    tokens_generated=len(result.log.events()),
                    switch_count=result.switch_count,
                    text=result.text,
                )
            )
    return rows


def rows_to_dicts(rows: list[ComparisonRow]) -> list[dict]:
    return [asdict(row) for row in rows]
