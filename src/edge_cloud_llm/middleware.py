"""Middleware Core: the token-by-token generation loop that ties everything
together — confidence extraction, the decision policy, and switching between
edge/cloud backends at sentence boundaries with a fallback to edge-only if
the cloud backend fails (sections 4-6 of CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from .communication import is_switch_boundary
from .confidence import ConfidenceExtractor
from .fault_tolerance import CloudUnavailableError, run_with_timeout
from .model_backend import ModelBackend
from .observability import EventLog, StepEvent
from .policy import DecisionPolicy, Route

DEFAULT_MAX_NEW_TOKENS = 64


@dataclass
class GenerationResult:
    text: str
    switch_count: int
    cloud_fallback_count: int
    log: EventLog


class MiddlewareCore:
    def __init__(
        self,
        edge_backend: ModelBackend,
        cloud_backend: ModelBackend,
        confidence_extractor: ConfidenceExtractor,
        policy: DecisionPolicy,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        self._edge = edge_backend
        self._cloud = cloud_backend
        self._confidence = confidence_extractor
        self._policy = policy
        self._max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> GenerationResult:
        self._policy.reset()
        log = EventLog()

        current_route = Route.EDGE
        current_backend = self._edge
        result = current_backend.prefill(prompt)

        token_ids = [result.token_id]
        generated_text = current_backend.decode(token_ids)

        score = self._confidence.score(result.logits)
        desired_route = self._policy.observe(score)
        log.record(StepEvent(0, current_route.value, result.token_id, score, switched=False))

        step = 1
        while step < self._max_new_tokens and token_ids[-1] != current_backend.eos_token_id():
            last_token_text = current_backend.decode([token_ids[-1]])
            can_switch = desired_route != current_route and is_switch_boundary(last_token_text)
            switched = False
            cloud_fallback = False

            if can_switch and desired_route is Route.CLOUD:
                context_text = prompt + generated_text
                try:
                    result = run_with_timeout(lambda: self._cloud.prefill(context_text))
                    current_backend, current_route, switched = self._cloud, Route.CLOUD, True
                except CloudUnavailableError:
                    result = current_backend.next_token(result.state)
                    cloud_fallback = True

            elif can_switch and desired_route is Route.EDGE:
                context_text = prompt + generated_text
                result = self._edge.prefill(context_text)
                current_backend, current_route, switched = self._edge, Route.EDGE, True

            elif current_route is Route.CLOUD:
                try:
                    result = run_with_timeout(lambda: current_backend.next_token(result.state))
                except CloudUnavailableError:
                    result = self._edge.prefill(prompt + generated_text)
                    current_backend, current_route = self._edge, Route.EDGE
                    cloud_fallback = True

            else:
                result = current_backend.next_token(result.state)

            token_ids.append(result.token_id)
            generated_text += current_backend.decode([result.token_id])

            score = self._confidence.score(result.logits)
            desired_route = self._policy.observe(score)
            log.record(
                StepEvent(step, current_route.value, result.token_id, score, switched, cloud_fallback)
            )
            step += 1

        return GenerationResult(
            text=generated_text,
            switch_count=log.switch_count(),
            cloud_fallback_count=log.cloud_fallback_count(),
            log=log,
        )
