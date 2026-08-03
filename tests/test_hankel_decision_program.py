from __future__ import annotations

import math

import torch

from vortex_runtime.hankel_decision_program import (
    build_hankel_feature,
    fit_hankel_decision_program,
    hankel_feature_size,
    hankel_program_budget,
    reduce_hidden,
    rollout_hankel_program,
)


def test_feature_sizes_include_requested_lifts() -> None:
    assert hankel_feature_size(
        state_rank=8,
        control_rank=4,
        order=2,
        lift="linear",
    ) == 21
    assert hankel_feature_size(
        state_rank=8,
        control_rank=4,
        order=2,
        lift="quadratic",
    ) == 29
    assert hankel_feature_size(
        state_rank=8,
        control_rank=4,
        order=2,
        lift="bilinear",
    ) == 29
    assert hankel_feature_size(
        state_rank=8,
        control_rank=4,
        order=2,
        lift="full",
    ) == 37


def test_405b_rank32_budget_matches_certificate() -> None:
    budget = hankel_program_budget(
        state_rank=32,
        control_rank=16,
        order=2,
        lift="full",
    )
    assert budget.feature_size == 145
    assert math.isclose(budget.total_program_gib, 0.006738841533660889)
    assert math.isclose(budget.hot_compute_gflop_per_token, 0.008217664)
    assert 21.0 < budget.minimum_build_reuse_tokens < 21.1


def test_build_feature_uses_history_control_and_lifts() -> None:
    history = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    control = torch.tensor([5.0])
    feature = build_hankel_feature(
        history,
        control,
        order=2,
        lift="full",
    )
    assert feature.tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        1.0,
        4.0,
        5.0,
        0.0,
        1.0,
    ]


def _synthetic_trajectory() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    hidden_size = 6
    vocabulary = 7
    state_rank = 2
    control_rank = 2
    total = 72

    state_basis, _ = torch.linalg.qr(torch.randn(hidden_size, state_rank))
    control_basis, _ = torch.linalg.qr(torch.randn(hidden_size, control_rank))
    control_coordinates = torch.randn(vocabulary, control_rank)
    embedding = control_coordinates @ control_basis.T
    token_ids = torch.tensor([(index * 3 + 1) % vocabulary for index in range(total)])
    transition = torch.tensor([[0.82, 0.12], [-0.07, 0.76]])
    control_map = torch.tensor([[0.14, -0.05], [0.03, 0.11]])
    mean = torch.randn(hidden_size) * 0.2

    reduced = [torch.tensor([0.25, -0.15])]
    for index in range(total - 1):
        next_control = control_coordinates[token_ids[index + 1]]
        reduced.append(transition @ reduced[-1] + control_map @ next_control)
    reduced_tensor = torch.stack(reduced)
    hidden = mean + reduced_tensor @ state_basis.T
    lm_head = torch.randn(vocabulary, hidden_size)
    return hidden, token_ids, embedding, lm_head


def test_linear_program_recovers_controlled_synthetic_dynamics() -> None:
    hidden, tokens, embedding, lm_head = _synthetic_trajectory()
    build_count = 52
    program = fit_hankel_decision_program(
        hidden_states=hidden[:build_count],
        token_ids=tokens[:build_count],
        embedding_weight=embedding,
        lm_head_weight=lm_head,
        lm_head_bias=None,
        state_rank=2,
        control_rank=2,
        order=1,
        lift="linear",
        ridge=1e-8,
    )
    assert program.diagnostics.training_relative_l2_error < 1e-4

    initial = [reduce_hidden(program, hidden[build_count - 1])]
    future_controls = tokens[build_count : build_count + 12].tolist()
    rollout = rollout_hankel_program(
        program,
        initial_history=initial,
        first_control_token=future_controls[0],
        forced_control_tokens=future_controls,
        steps=len(future_controls),
    )
    expected = hidden[build_count : build_count + len(future_controls)]
    relative = torch.linalg.vector_norm(rollout.hidden_states - expected) / torch.linalg.vector_norm(expected)
    assert float(relative.item()) < 1e-3


def test_invalid_lift_is_rejected() -> None:
    try:
        hankel_feature_size(
            state_rank=4,
            control_rank=2,
            order=1,
            lift="unknown",
        )
    except ValueError as error:
        assert "lift" in str(error)
    else:
        raise AssertionError("invalid lift should fail")
