"""Step 5 + 6: a sample request runs start to finish through the middleware
loop with a mid-response switch, and a simulated cloud outage falls back to
the edge instead of crashing."""

from edge_cloud_llm.confidence import ConfidenceExtractor, ConfidenceMetric
from edge_cloud_llm.middleware import MiddlewareCore
from edge_cloud_llm.policy import DecisionPolicy

from fakes import FailingCloudBackend, FakeBackend


def _middleware(cloud_backend):
    edge = FakeBackend("edge", [(0, True), (1, True), (2, False), (4, True)])
    confidence = ConfidenceExtractor(metric=ConfidenceMetric.MARGIN)
    policy = DecisionPolicy(threshold=0.5, window_size=1)
    return MiddlewareCore(edge, cloud_backend, confidence, policy, max_new_tokens=4)


def test_sample_request_completes_and_switches_to_cloud():
    cloud = FakeBackend("cloud", [(3, True)])
    middleware = _middleware(cloud)

    result = middleware.generate("prompt")

    assert result.text == "ab.c"
    assert result.switch_count == 1
    assert any(event.route == "cloud" for event in result.log.events())


def test_simulated_cloud_outage_falls_back_to_edge_without_crashing():
    cloud = FailingCloudBackend("cloud", [])
    middleware = _middleware(cloud)

    result = middleware.generate("prompt")

    assert result.text == "ab.d"
    assert result.switch_count == 0
    assert all(event.route == "edge" for event in result.log.events())
