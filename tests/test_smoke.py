"""Step 1 smoke test: project is importable and pytest runs green."""

import edge_cloud_llm


def test_package_importable():
    assert edge_cloud_llm is not None
