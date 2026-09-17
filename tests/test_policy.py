"""Step 4: decision policy switches correctly near the threshold, and can
flip back and forth within a single response (section 3's core novelty)."""

from edge_cloud_llm.policy import DecisionPolicy, Route


def test_stays_on_edge_when_confidence_high():
    policy = DecisionPolicy(threshold=0.5, window_size=4)
    for score in [0.9, 0.85, 0.8, 0.95]:
        route = policy.observe(score)
    assert route == Route.EDGE


def test_switches_to_cloud_when_window_average_drops_below_threshold():
    policy = DecisionPolicy(threshold=0.5, window_size=4)
    for score in [0.9, 0.9, 0.9, 0.9]:
        policy.observe(score)
    route = None
    for score in [0.1, 0.1, 0.1, 0.1]:
        route = policy.observe(score)
    assert route == Route.CLOUD


def test_boundary_exactly_at_threshold_counts_as_edge():
    policy = DecisionPolicy(threshold=0.5, window_size=1)
    assert policy.observe(0.5) == Route.EDGE
    assert policy.observe(0.4999) == Route.CLOUD


def test_can_switch_back_and_forth_within_one_response():
    policy = DecisionPolicy(threshold=0.5, window_size=1)
    routes = [policy.observe(s) for s in [0.9, 0.1, 0.9]]
    assert routes == [Route.EDGE, Route.CLOUD, Route.EDGE]
