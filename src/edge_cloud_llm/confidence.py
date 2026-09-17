"""Confidence Extractor: per-token confidence score straight from softmax.

Section 7 of CLAUDE.md: three cheap metrics, no extra forward pass, plus a
temperature-scaling calibration step so scores are comparable across models
of different sizes (edge vs. cloud).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


class ConfidenceMetric(str, Enum):
    MAX_PROB = "max_prob"
    ENTROPY = "entropy"
    MARGIN = "margin"


@dataclass
class Calibration:
    """Per-backend temperature used to make confidence scores comparable."""

    temperature: float = 1.0


def _softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=-1)


def max_prob_confidence(logits: torch.Tensor, temperature: float = 1.0) -> float:
    probs = _softmax(logits, temperature)
    return float(probs.max().item())


def entropy_confidence(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """Confidence in [0, 1]: 1 - normalized entropy (low entropy = high confidence)."""
    probs = _softmax(logits, temperature)
    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * log_probs).sum()
    max_entropy = torch.log(torch.tensor(float(probs.shape[-1])))
    normalized = (entropy / max_entropy).item()
    return float(1.0 - normalized)


def margin_confidence(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """Margin between the top-2 softmax probabilities, used by the closest
    known baseline (arXiv:2602.07958)."""
    probs = _softmax(logits, temperature)
    top2 = torch.topk(probs, k=2, dim=-1).values
    return float((top2[0] - top2[1]).item())


_METRIC_FUNCTIONS = {
    ConfidenceMetric.MAX_PROB: max_prob_confidence,
    ConfidenceMetric.ENTROPY: entropy_confidence,
    ConfidenceMetric.MARGIN: margin_confidence,
}


@dataclass
class ConfidenceExtractor:
    metric: ConfidenceMetric = ConfidenceMetric.MARGIN
    calibration: Calibration = field(default_factory=Calibration)

    def score(self, logits: torch.Tensor) -> float:
        fn = _METRIC_FUNCTIONS[self.metric]
        return fn(logits, self.calibration.temperature)
