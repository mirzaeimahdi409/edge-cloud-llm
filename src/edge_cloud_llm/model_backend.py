"""Model execution port: one uniform interface for the edge and cloud backends.

Section 6 of CLAUDE.md requires the middleware to make no assumption about the
model type. Any backend that implements ``ModelBackend`` (prefill/next_token/name)
can be plugged in without touching the rest of the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class KVState:
    """Everything a backend needs to produce the next token."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: object | None


@dataclass
class NextTokenResult:
    token_id: int
    logits: torch.Tensor
    state: KVState


@runtime_checkable
class ModelBackend(Protocol):
    def prefill(self, context: str) -> NextTokenResult:
        """Run the given text through the model and return the first token
        that follows it, together with the state needed to continue."""

    def next_token(self, state: KVState) -> NextTokenResult:
        """Produce one token given the current state."""

    def decode(self, token_ids: list[int]) -> str:
        """Turn generated token ids back into text."""

    def name(self) -> str:
        ...

    def eos_token_id(self) -> int:
        ...


class HFModelBackend:
    """ModelBackend implementation on top of a Hugging Face causal LM.

    Re-prefill on switch (section 8) works because edge and cloud backends
    share the same tokenizer family: text, not KV-cache tensors, is what
    crosses the edge/cloud boundary.
    """

    def __init__(self, model_name: str, label: str, device: str = "cpu"):
        self._label = label
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._model.to(device)
        self._model.eval()

    def name(self) -> str:
        return self._label

    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.no_grad()
    def prefill(self, context: str) -> NextTokenResult:
        encoded = self._tokenizer(context, return_tensors="pt").to(self._device)
        outputs = self._model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]
        token_id = int(logits.argmax(dim=-1).item())
        state = KVState(
            input_ids=torch.tensor([[token_id]], device=self._device),
            attention_mask=encoded["attention_mask"],
            past_key_values=outputs.past_key_values,
        )
        return NextTokenResult(token_id=token_id, logits=logits[0], state=state)

    @torch.no_grad()
    def next_token(self, state: KVState) -> NextTokenResult:
        attention_mask = torch.cat(
            [state.attention_mask, torch.ones_like(state.input_ids)], dim=-1
        )
        outputs = self._model(
            input_ids=state.input_ids,
            attention_mask=attention_mask,
            past_key_values=state.past_key_values,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]
        token_id = int(logits.argmax(dim=-1).item())
        next_state = KVState(
            input_ids=torch.tensor([[token_id]], device=self._device),
            attention_mask=attention_mask,
            past_key_values=outputs.past_key_values,
        )
        return NextTokenResult(token_id=token_id, logits=logits[0], state=next_state)


# Section 15 default model choice: same open-source family (Qwen2.5), a small
# instruct model for the edge and a larger one from the same family for the
# cloud, so tokenizer/text-based re-prefill (section 8) is always valid.
EDGE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CLOUD_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class EdgeModelAdapter(HFModelBackend):
    def __init__(self, device: str = "cpu"):
        super().__init__(EDGE_MODEL_NAME, label="edge", device=device)


class CloudModelAdapter(HFModelBackend):
    def __init__(self, device: str = "cpu"):
        super().__init__(CLOUD_MODEL_NAME, label="cloud", device=device)
