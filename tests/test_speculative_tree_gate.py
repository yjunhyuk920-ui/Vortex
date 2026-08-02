from __future__ import annotations

from vortex_runtime.feasibility import ModelSpec
from vortex_runtime.speculative_tree_gate import (
    longest_reference_prefix,
    speculative_tree_verification_budget,
    unique_prefix_node_count,
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


def test_unique_prefix_nodes_share_common_roots() -> None:
    sequences = [(1, 2, 3), (1, 2, 4), (1, 5, 6)]
    assert unique_prefix_node_count(sequences) == 7


def test_longest_reference_prefix_across_branches() -> None:
    sequences = [(1, 2, 9), (1, 2, 3), (1, 8, 8)]
    assert longest_reference_prefix(sequences, (1, 2, 3, 4)) == 3


def test_tree_depth_must_cover_required_commitment() -> None:
    target = _model(405_000_000_000, weight_bits=16)
    baseline = _model(4_000_000_000, weight_bits=4)
    budget = speculative_tree_verification_budget(
        target=target,
        baseline=baseline,
        hot_bits=4,
        tree_nodes=1024,
        tree_depth=16,
        committed_tokens=8,
        host_to_device_gib_s=24.0,
        hot_effective_tops=160.0,
    )
    assert budget.minimum_committed_tokens_ideal > budget.tree_depth
    assert not budget.depth_can_meet_ideal
    assert not budget.observed_ideal_pass
    assert budget.hot_weight_gib > 180
