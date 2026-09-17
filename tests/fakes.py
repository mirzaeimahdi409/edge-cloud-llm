"""Deterministic fake ModelBackend for fast, offline middleware tests."""

from __future__ import annotations

import torch

from edge_cloud_llm.model_backend import KVState, NextTokenResult

EOS_TOKEN_ID = -1

# Token ids are just indices into this vocabulary; "." marks a switch boundary.
_VOCAB = {0: "a", 1: "b", 2: ".", 3: "c", 4: "d"}


def _logits_for(token_id: int, confident: bool) -> torch.Tensor:
    logits = torch.full((5,), -10.0)
    if confident:
        logits[token_id] = 10.0
    else:
        # Near-tie with a neighbor -> low margin confidence.
        logits[token_id] = 1.0
        logits[(token_id + 1) % 5] = 0.9
    return logits


class FakeBackend:
    """Plays back a fixed script of (token_id, confident) pairs, looping the
    last entry once the script is exhausted (e.g. after a switch)."""

    def __init__(self, label: str, script: list[tuple[int, bool]]):
        self._label = label
        self._script = script
        self._position = 0

    def name(self) -> str:
        return self._label

    def eos_token_id(self) -> int:
        return EOS_TOKEN_ID

    def decode(self, token_ids: list[int]) -> str:
        return "".join(_VOCAB[t] for t in token_ids)

    def _next_scripted(self) -> tuple[int, bool]:
        entry = self._script[min(self._position, len(self._script) - 1)]
        self._position += 1
        return entry

    def prefill(self, context: str) -> NextTokenResult:
        token_id, confident = self._next_scripted()
        state = KVState(input_ids=torch.tensor([[token_id]]), attention_mask=torch.ones(1, 1), past_key_values=None)
        return NextTokenResult(token_id=token_id, logits=_logits_for(token_id, confident), state=state)

    def next_token(self, state: KVState) -> NextTokenResult:
        token_id, confident = self._next_scripted()
        new_state = KVState(input_ids=torch.tensor([[token_id]]), attention_mask=state.attention_mask, past_key_values=None)
        return NextTokenResult(token_id=token_id, logits=_logits_for(token_id, confident), state=new_state)


class FailingCloudBackend(FakeBackend):
    def prefill(self, context: str) -> NextTokenResult:
        raise RuntimeError("simulated cloud outage")

    def next_token(self, state: KVState) -> NextTokenResult:
        raise RuntimeError("simulated cloud outage")
