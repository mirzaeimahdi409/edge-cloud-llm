"""τ/k sweep produces one row per (prompt, threshold, window_size) combo."""

from edge_cloud_llm.confidence import ConfidenceExtractor, ConfidenceMetric
from edge_cloud_llm.evaluation import run_sweep

from fakes import FakeBackend


def test_sweep_covers_full_grid_without_crashing():
    edge = FakeBackend("edge", [(0, True), (1, True), (2, False), (4, True)])
    cloud = FakeBackend("cloud", [(3, True)])
    confidence = ConfidenceExtractor(metric=ConfidenceMetric.MARGIN)

    rows = run_sweep(
        prompts=["p1", "p2"],
        thresholds=(0.2, 0.8),
        window_sizes=(1, 2),
        max_new_tokens=4,
        edge=edge,
        cloud=cloud,
        confidence=confidence,
    )

    assert len(rows) == 2 * 2 * 2
    seen = {(row.prompt, row.threshold, row.window_size) for row in rows}
    assert seen == {
        (prompt, threshold, window_size)
        for prompt in ["p1", "p2"]
        for threshold in (0.2, 0.8)
        for window_size in (1, 2)
    }
    assert all(len(row.text) > 0 for row in rows)
