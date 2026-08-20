from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vortex_runtime.prefix_state_bisimulation import (
    PrefixBisimulationError,
    PrefixRNGState,
    append_exact_token,
    cache_sha256,
    first_byte_mismatch,
    initialize_prefix_state,
    replay_prefix,
    select_token,
    tensor_bytes,
)


@dataclass
class FakeOutput:
    logits: torch.Tensor
    past_key_values: Any


class FakeCausalModel:
    """Small deterministic cache-bearing model for replay contract tests."""

    vocab_size = 17

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = True,
        return_dict: bool = True,
    ) -> FakeOutput:
        del use_cache, return_dict
        incoming = input_ids.detach().to(torch.int64).reshape(-1)
        if past_key_values is None:
            tokens = incoming
        else:
            previous = past_key_values[0][0].reshape(-1).to(torch.int64)
            tokens = torch.cat([previous, incoming], dim=0)
        key = tokens.to(torch.float32).reshape(1, 1, -1, 1)
        value = (tokens * 3 + 1).to(torch.float32).reshape(1, 1, -1, 1)
        cache = ((key, value),)
        center = int((tokens.sum().item() + 5 * tokens.numel()) % self.vocab_size)
        logits = -torch.arange(self.vocab_size, dtype=torch.float32).abs()
        logits = torch.roll(logits, shifts=center).reshape(1, 1, -1)
        return FakeOutput(logits=logits, past_key_values=cache)


def test_prefix_digest_changes_when_exact_token_is_appended() -> None:
    rng = PrefixRNGState.greedy()
    state = initialize_prefix_state([1, 2, 3], rng)
    next_state = append_exact_token(state, 4, rng)
    assert state.digest() != next_state.digest()
    assert next_state.prefix_ids == (1, 2, 3, 4)
    assert next_state.step == 1


def test_empty_prompt_and_invalid_state_fail_closed() -> None:
    with pytest.raises(PrefixBisimulationError):
        initialize_prefix_state([], PrefixRNGState.greedy())
    with pytest.raises(PrefixBisimulationError):
        initialize_prefix_state([1, -2], PrefixRNGState.greedy())
    with pytest.raises(PrefixBisimulationError):
        PrefixRNGState.sampled(1, temperature=0.0)
    with pytest.raises(PrefixBisimulationError):
        PrefixRNGState.sampled(1, top_k=-1)


def test_greedy_selection_preserves_rng() -> None:
    logits = torch.tensor([0.0, 2.0, 1.0], dtype=torch.float32)
    rng = PrefixRNGState.greedy()
    token, next_rng = select_token(logits, rng)
    assert token == 1
    assert next_rng == rng


def test_sampled_selection_replays_from_identical_rng_bytes() -> None:
    logits = torch.tensor([0.1, 0.4, 2.0, 1.7, -0.3], dtype=torch.float32)
    rng = PrefixRNGState.sampled(20260820, temperature=0.8, top_k=3)
    token_a, next_a = select_token(logits, rng)
    token_b, next_b = select_token(logits.clone(), rng)
    assert token_a == token_b
    assert next_a.digest() == next_b.digest()
    assert next_a.generator_state != rng.generator_state


def test_one_byte_cache_change_changes_digest_and_is_located() -> None:
    left = ((torch.tensor([[[[1.0], [2.0]]]], dtype=torch.float32),),)
    right_tensor = left[0][0].clone()
    right_tensor.view(torch.uint8).reshape(-1)[0] ^= 0x01
    right = ((right_tensor,),)
    assert cache_sha256(left) != cache_sha256(right)
    mismatch = first_byte_mismatch(tensor_bytes(left[0][0]), tensor_bytes(right_tensor))
    assert mismatch is not None
    assert mismatch["offset"] == 0


def test_replay_prefix_matches_incremental_fake_reference() -> None:
    model = FakeCausalModel()
    rng = PrefixRNGState.greedy()
    state = initialize_prefix_state([2, 5, 7], rng)

    reference = model(
        input_ids=torch.tensor([[2, 5, 7]], dtype=torch.long),
        use_cache=True,
        return_dict=True,
    )
    for expected_step in range(4):
        replay = replay_prefix(model, state)
        reference_logits = reference.logits[:, -1, :]
        assert tensor_bytes(replay.logits) == tensor_bytes(reference_logits)
        assert replay.cache_sha256 == cache_sha256(reference.past_key_values)

        token_ref, next_rng_ref = select_token(reference_logits, state.rng_state)
        token_replay, next_rng_replay = select_token(replay.logits, state.rng_state)
        assert token_ref == token_replay
        assert next_rng_ref.digest() == next_rng_replay.digest()
        state = append_exact_token(state, token_replay, next_rng_replay)
        reference = model(
            input_ids=torch.tensor([[token_ref]], dtype=torch.long),
            past_key_values=reference.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        assert state.step == expected_step + 1


def test_replay_accounting_grows_with_prefix() -> None:
    model = FakeCausalModel()
    state = initialize_prefix_state([1, 2, 3, 4], PrefixRNGState.greedy())
    first = replay_prefix(model, state)
    assert first.target_calls == 1
    assert first.replayed_token_positions == 4

    state = append_exact_token(state, 5, PrefixRNGState.greedy())
    state = append_exact_token(state, 6, PrefixRNGState.greedy())
    later = replay_prefix(model, state)
    assert later.target_calls == 3
    assert later.replayed_token_positions == 6
