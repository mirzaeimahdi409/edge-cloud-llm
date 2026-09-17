"""Step 8: the middleware runs inside an EdgeSimPy scenario without error."""

from edge_cloud_llm.confidence import ConfidenceExtractor, ConfidenceMetric
from edge_cloud_llm.edgesim_scenario import NetworkScenario, run_scenario
from edge_cloud_llm.middleware import MiddlewareCore
from edge_cloud_llm.policy import DecisionPolicy

from fakes import FakeBackend


def test_scenario_runs_without_error():
    edge = FakeBackend("edge", [(0, True), (1, True), (2, False)])
    cloud = FakeBackend("cloud", [(3, True)])
    middleware = MiddlewareCore(
        edge, cloud, ConfidenceExtractor(metric=ConfidenceMetric.MARGIN), DecisionPolicy(threshold=0.5, window_size=1), max_new_tokens=4
    )

    generations = run_scenario(
        middleware=middleware,
        prompt="prompt",
        scenario=NetworkScenario(bandwidth_mbps=100.0, delay_ms=20.0),
        num_ticks=2,
        logs_directory="/tmp/edge_cloud_llm_scenario_logs",
    )

    assert len(generations) == 2
    assert all(len(g.text) > 0 for g in generations)
