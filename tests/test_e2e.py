"""Step 7 end-to-end test with the real edge/cloud models: one full request
that never switches, and one full request that does switch to the cloud."""

from edge_cloud_llm.confidence import ConfidenceExtractor, ConfidenceMetric
from edge_cloud_llm.middleware import MiddlewareCore
from edge_cloud_llm.model_backend import CloudModelAdapter, EdgeModelAdapter
from edge_cloud_llm.policy import DecisionPolicy

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 24


def _middleware(threshold: float) -> MiddlewareCore:
    return MiddlewareCore(
        edge_backend=EdgeModelAdapter(),
        cloud_backend=CloudModelAdapter(),
        confidence_extractor=ConfidenceExtractor(metric=ConfidenceMetric.MARGIN),
        policy=DecisionPolicy(threshold=threshold, window_size=4),
        max_new_tokens=MAX_NEW_TOKENS,
    )


def test_full_request_without_switch_completes():
    middleware = _middleware(threshold=-1.0)  # margin is always >= 0 -> never below threshold
    result = middleware.generate(PROMPT)

    assert len(result.text.strip()) > 0
    assert result.switch_count == 0
    assert all(event.route == "edge" for event in result.log.events())


def test_full_request_with_switch_completes():
    middleware = _middleware(threshold=1.1)  # margin is always <= 1 -> always below threshold
    result = middleware.generate(PROMPT)

    assert len(result.text.strip()) > 0
    assert result.switch_count >= 1
    assert any(event.route == "cloud" for event in result.log.events())
