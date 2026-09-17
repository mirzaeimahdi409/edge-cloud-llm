"""Decision Policy Engine: segment-level switch decision, not request-level.

Section 7/12 of CLAUDE.md: a fixed-threshold policy over a sliding window of
the last k token confidence scores. This is the core novelty — the decision
is re-evaluated continuously during a single response, not once at the start
of the request (section 2/3), and it can flip back and forth multiple times.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum


class Route(str, Enum):
    EDGE = "edge"
    CLOUD = "cloud"


@dataclass
class DecisionPolicy:
    threshold: float = 0.5
    window_size: int = 6
    _window: deque[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._window = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self._window.clear()

    def observe(self, confidence: float) -> Route:
        """Feed one token's confidence score, return the route for what
        follows: EDGE while the recent window stays above threshold, CLOUD
        as soon as the windowed average drops below it."""
        self._window.append(confidence)
        window_average = sum(self._window) / len(self._window)
        return Route.EDGE if window_average >= self.threshold else Route.CLOUD
