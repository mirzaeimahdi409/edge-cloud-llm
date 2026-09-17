"""Edge-Cloud Communication Layer: how context crosses the boundary on switch.

Section 8 of CLAUDE.md: KV-cache cannot move between two different backend
instances, so what actually crosses the boundary is the generated *text*
(re-prefill), and only at a sentence/clause boundary to keep re-prefill
overhead bounded.
"""

from __future__ import annotations

_BOUNDARY_CHARS = {".", "!", "?", "،", "؛", ";", "\n"}


def is_switch_boundary(token_text: str) -> bool:
    """Whether the given piece of decoded text marks a sentence/clause
    boundary, i.e. a point where switching backends is allowed."""
    return any(char in token_text for char in _BOUNDARY_CHARS)
