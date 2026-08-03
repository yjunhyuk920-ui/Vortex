from __future__ import annotations

import math

import torch

from vortex_runtime.final_hidden_trace import (
    PromptContinuationTrace,
    continuation_queries_after_anchor,
)
from vortex_runtime.nonlocal_decision_memory import (
    build_nonlocal_decision_memory,
    decision_memory_budget,
    evaluate_nonlocal_decision_memory,
    prefix_lengths_for_entries,
)


def test_405b_scaled_budget_is_small_and_explicit() -> None:
    budget = decision_memory_budget(
        entries=65536,
        hidden_size=16384,
        key_rank=128,
        block_length=256,
    )
    assert math.isclose(budget.keys_gib, 0.015625, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(budget.blocks_gib, 0.0625, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(budget.index_gib, 0.01953125, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(budget.total_gib, 0.09765625, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(
        budget.total_lookup_gflop_per_query,
        0.02097152,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert budget.memory_pass
    assert budget.lookup_pass


def test_evaluation_starts_after_one_exact_boundary_anchor() -> None:
    trace = PromptContinuationTrace(
        prompt="synthetic",
        prompt_token_ids=torch.tensor([10, 20]),
        prompt_hidden_states=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        first_generated_token=99,
        continuation_token_ids=torch.tensor([99, 30, 40, 50]),
        continuation_hidden_states=torch.tensor(
            [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
        ),
    )
    queries, targets = continuation_queries_after_anchor(trace, steps=3)
    assert queries.tolist() == [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
    assert targets.tolist() == [30, 40, 50]
    assert int(trace.continuation_token_ids[0].item()) == trace.first_generated_token


def test_memory_contains_prompt_only_following_blocks() -> None:
    hidden = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    tokens = torch.tensor([10, 11, 12, 13])
    memory = build_nonlocal_decision_memory(
        prompt_hidden_states=hidden,
        prompt_token_ids=tokens,
        key_rank=2,
        block_length=4,
    )
    assert memory.entries == 3
    assert memory.positions.tolist() == [0, 1, 2]
    assert memory.lengths.tolist() == [3, 2, 1]
    assert memory.blocks.tolist() == [
        [11, 12, 13, -1],
        [12, 13, -1, -1],
        [13, -1, -1, -1],
    ]


def test_prefix_length_stops_at_first_mismatch_and_entry_end() -> None:
    torch.manual_seed(3)
    hidden = torch.randn(5, 6)
    tokens = torch.tensor([1, 2, 3, 4, 5])
    memory = build_nonlocal_decision_memory(
        prompt_hidden_states=hidden,
        prompt_token_ids=tokens,
        key_rank=3,
        block_length=4,
    )
    prefixes = prefix_lengths_for_entries(
        memory,
        target_suffix=torch.tensor([3, 4, 99, 5]),
    )
    assert prefixes.tolist() == [0, 2, 0, 0]


def test_global_oracle_is_an_upper_bound_on_hidden_retrieval() -> None:
    torch.manual_seed(7)
    hidden = torch.randn(7, 8)
    tokens = torch.tensor([10, 20, 30, 40, 50, 60, 70])
    memory = build_nonlocal_decision_memory(
        prompt_hidden_states=hidden,
        prompt_token_ids=tokens,
        key_rank=4,
        block_length=4,
    )
    query = hidden[0:1]
    # Hidden retrieval should prefer prompt position zero, whose block starts 20.
    # The impossible global token oracle can instead choose position three,
    # whose prompt-only block is 50, 60, 70.
    frontier = evaluate_nonlocal_decision_memory(
        memory,
        query_hidden_states=query,
        continuation_token_ids=torch.tensor([50, 60, 70, 99]),
        topk_values=(2, 4),
        scaled_entries=64,
        target_hidden_size=8,
    )
    assert frontier.nearest.first == 0
    assert frontier.global_oracle.first == 3
    assert frontier.global_oracle.maximum >= frontier.nearest.maximum


def test_exact_nearest_state_can_replay_a_prompt_block() -> None:
    torch.manual_seed(11)
    hidden = torch.randn(8, 10)
    tokens = torch.tensor([1, 4, 9, 16, 25, 36, 49, 64])
    memory = build_nonlocal_decision_memory(
        prompt_hidden_states=hidden,
        prompt_token_ids=tokens,
        key_rank=6,
        block_length=5,
    )
    frontier = evaluate_nonlocal_decision_memory(
        memory,
        query_hidden_states=hidden[2:3],
        continuation_token_ids=torch.tensor([16, 25, 36, 49, 64]),
        topk_values=(1, 4),
        scaled_entries=64,
        target_hidden_size=10,
    )
    assert frontier.nearest.first == 5
    assert frontier.nearest_positions[0] == 2
    assert frontier.topk_oracles[1].first == 5
    assert frontier.global_oracle.first == 5
