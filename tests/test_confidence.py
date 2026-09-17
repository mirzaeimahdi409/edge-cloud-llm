"""Step 3: confidence metrics computed correctly on fake logits."""

import math

import torch

from edge_cloud_llm.confidence import (
    ConfidenceExtractor,
    ConfidenceMetric,
    entropy_confidence,
    margin_confidence,
    max_prob_confidence,
)


def test_max_prob_confident_when_one_token_dominates():
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    assert max_prob_confidence(logits) > 0.99


def test_max_prob_low_when_uniform():
    logits = torch.zeros(4)
    assert math.isclose(max_prob_confidence(logits), 0.25, abs_tol=1e-6)


def test_entropy_confidence_high_for_peaked_distribution():
    peaked = torch.tensor([10.0, 0.0, 0.0, 0.0])
    uniform = torch.zeros(4)
    assert entropy_confidence(peaked) > entropy_confidence(uniform)
    assert math.isclose(entropy_confidence(uniform), 0.0, abs_tol=1e-6)


def test_margin_confidence_zero_when_tied():
    logits = torch.tensor([1.0, 1.0, 0.0])
    assert math.isclose(margin_confidence(logits), 0.0, abs_tol=1e-6)


def test_margin_confidence_high_when_separated():
    logits = torch.tensor([10.0, -10.0, -10.0])
    assert margin_confidence(logits) > 0.99


def test_temperature_scaling_lowers_confidence():
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    extractor_cold = ConfidenceExtractor(metric=ConfidenceMetric.MAX_PROB)
    extractor_cold.calibration.temperature = 1.0
    extractor_hot = ConfidenceExtractor(metric=ConfidenceMetric.MAX_PROB)
    extractor_hot.calibration.temperature = 5.0
    assert extractor_hot.score(logits) < extractor_cold.score(logits)
