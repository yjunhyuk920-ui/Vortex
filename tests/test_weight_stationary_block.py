from __future__ import annotations

import torch

from vortex_runtime.feasibility import ModelSpec
from vortex_runtime.weight_stationary_block import (
    StreamedBlockHardware,
    certified_fixed_prefix,
    jacobi_token_update,
    longest_common_prefix,
    streamed_exact_block_budget,
)


def _model(parameters: int, *, weight_bits: int) -> ModelSpec:
    return ModelSpec(
        parameters=parameters,
        layers=4,
        hidden_size=128,
        intermediate_size=256,
        attention_heads=8,
        kv_heads=2,
        vocab_size=1024,
        context_tokens=64,
        weight_bits=weight_bits,
        kv_bits=4,
    )


def test_jacobi_update_shifts_parallel_predictions() -> None:
    draft = torch.tensor([[9, 9, 9, 9]])
    logits = torch.full((1, 4, 16), -100.0)
    logits[0, 0, 2] = 1.0
    logits[0, 1, 3] = 1.0
    logits[0, 2, 4] = 1.0
    logits[0, 3, 5] = 1.0
    updated = jacobi_token_update(
        draft_tokens=draft,
        prompt_next_token=torch.tensor([1]),
        draft_logits=logits,
    )
    assert updated.tolist() == [[1, 2, 3, 4]]


def test_prefix_certificate_and_common_prefix() -> None:
    previous = torch.tensor([[1, 2, 7, 8]])
    updated = torch.tensor([[1, 2, 3, 4]])
    assert certified_fixed_prefix(previous, updated) == 2
    assert longest_common_prefix(updated, torch.tensor([[1, 2, 3, 9]])) == 3


def test_roofline_budget_rewards_committed_block_tokens() -> None:
    target = _model(405_000_000_000, weight_bits=4)
    baseline = _model(4_000_000_000, weight_bits=4)
    hardware = StreamedBlockHardware(
        host_to_device_gib_s=24.0,
        target_tensor_tflops=80.0,
        baseline_gpu_memory_gib_s=300.0,
        baseline_tensor_tflops=40.0,
    )
    short = streamed_exact_block_budget(
        target=target,
        baseline=baseline,
        draft_positions=64,
        committed_tokens=8,
        target_passes=4,
        hardware=hardware,
    )
    long = streamed_exact_block_budget(
        target=target,
        baseline=baseline,
        draft_positions=64,
        committed_tokens=64,
        target_passes=4,
        hardware=hardware,
    )
    assert long.ideal_seconds_per_committed_token < short.ideal_seconds_per_committed_token
    assert long.minimum_committed_tokens_ideal > 0
    assert long.target_weight_gib > 180
