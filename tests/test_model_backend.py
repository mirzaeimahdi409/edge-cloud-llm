"""Step 2: both adapters complete a sentence on their own via prefill + next_token."""

import pytest

from edge_cloud_llm.model_backend import CloudModelAdapter, EdgeModelAdapter

MAX_NEW_TOKENS = 15


def _generate(backend, prompt: str) -> str:
    result = backend.prefill(prompt)
    token_ids = [result.token_id]
    for _ in range(MAX_NEW_TOKENS - 1):
        if token_ids[-1] == backend.eos_token_id():
            break
        result = backend.next_token(result.state)
        token_ids.append(result.token_id)
    return backend.decode(token_ids)


@pytest.mark.parametrize("backend_cls", [EdgeModelAdapter, CloudModelAdapter])
def test_backend_completes_a_sentence(backend_cls):
    backend = backend_cls()
    text = _generate(backend, "The capital of France is")
    assert isinstance(text, str)
    assert len(text.strip()) > 0
