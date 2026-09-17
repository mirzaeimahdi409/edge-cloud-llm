#!/usr/bin/env python
"""Ad-hoc diagnostic: dump the full per-token event log for a few
(prompt, threshold, window_size) configs so the switching decisions and any
cloud fallbacks can be inspected directly, not just inferred from aggregate
CSV numbers."""

import sys
import time

from edge_cloud_llm.confidence import ConfidenceExtractor, ConfidenceMetric
from edge_cloud_llm.middleware import MiddlewareCore
from edge_cloud_llm.model_backend import CloudModelAdapter, EdgeModelAdapter
from edge_cloud_llm.policy import DecisionPolicy

CASES = [
    ("Water boils at a temperature of", 0.5, 4),
    ("The capital of France is", 0.5, 4),
    ("In 1969, humans first landed on", 0.8, 4),
    ("In 1969, humans first landed on", 0.8, 6),
]


def main() -> None:
    edge = EdgeModelAdapter()
    cloud = CloudModelAdapter()
    confidence = ConfidenceExtractor(metric=ConfidenceMetric.MARGIN)

    for prompt, threshold, window_size in CASES:
        middleware = MiddlewareCore(
            edge_backend=edge,
            cloud_backend=cloud,
            confidence_extractor=confidence,
            policy=DecisionPolicy(threshold=threshold, window_size=window_size),
            max_new_tokens=40,
        )
        start = time.perf_counter()
        result = middleware.generate(prompt)
        latency = time.perf_counter() - start

        print("=" * 100)
        print(f"prompt={prompt!r} threshold={threshold} window_size={window_size}")
        print(f"latency={latency:.2f}s switch_count={result.switch_count} cloud_fallback_count={result.cloud_fallback_count}")
        print(f"text={result.text!r}")
        print("step route      confidence  switched  cloud_fallback  token_id")
        for e in result.log.events():
            print(f"{e.step_index:4d} {e.route:10s} {e.confidence:10.4f}  {str(e.switched):8s} {str(e.cloud_fallback):14s} {e.token_id}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
