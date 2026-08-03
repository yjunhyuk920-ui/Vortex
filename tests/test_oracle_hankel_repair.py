from __future__ import annotations

import torch

from vortex_runtime.hankel_decision_program import (
    fit_hankel_decision_program,
    hankel_program_budget,
    reduce_hidden,
)
from vortex_runtime.oracle_hankel_repair import (
    TARGET_REPAIR_COMPUTE_GFLOP,
    TARGET_REPAIR_TRAFFIC_GIB,
    oracle_repair_hankel_rollout,
    repair_envelope_passes,
)


def _program_and_trajectory():
    torch.manual_seed(21)
    hidden_size = 5
    vocabulary = 6
    state_rank = 2
    control_rank = 2
    total = 48

    state_basis, _ = torch.linalg.qr(torch.randn(hidden_size, state_rank))
    control_basis, _ = torch.linalg.qr(torch.randn(hidden_size, control_rank))
    controls = torch.randn(vocabulary, control_rank)
    embedding = controls @ control_basis.T
    tokens = torch.tensor([(index * 5 + 2) % vocabulary for index in range(total)])
    transition = torch.tensor([[0.81, 0.09], [-0.04, 0.73]])
    control_map = torch.tensor([[0.13, -0.02], [0.04, 0.12]])
    mean = torch.randn(hidden_size) * 0.1

    reduced = [torch.tensor([0.2, -0.1])]
    for index in range(total - 1):
        reduced.append(transition @ reduced[-1] + control_map @ controls[tokens[index + 1]])
    hidden = mean + torch.stack(reduced) @ state_basis.T
    lm_head = torch.randn(vocabulary, hidden_size)
    program = fit_hankel_decision_program(
        hidden_states=hidden[:32],
        token_ids=tokens[:32],
        embedding_weight=embedding,
        lm_head_weight=lm_head,
        lm_head_bias=None,
        state_rank=2,
        control_rank=2,
        order=1,
        lift="linear",
        ridge=1e-8,
    )
    return program, hidden, tokens


def test_exact_controlled_dynamics_need_no_repairs_when_targets_follow_program() -> None:
    program, hidden, tokens = _program_and_trajectory()
    start = 31
    steps = 12
    initial = [reduce_hidden(program, hidden[start])]

    # Build exact targets from the program itself so this test isolates repair
    # accounting rather than the arbitrary synthetic LM head.
    history = list(initial)
    controls = tokens[start + 1 : start + 1 + steps]
    targets = []
    exact_hidden = []
    from vortex_runtime.hankel_decision_program import (
        decision_logits,
        predict_next_reduced,
        reconstruct_hidden,
    )

    for control in controls:
        next_state = predict_next_reduced(program, history, int(control.item()))
        targets.append(int(torch.argmax(decision_logits(program, next_state)).item()))
        exact_hidden.append(reconstruct_hidden(program, next_state))
        history = [next_state]

    rollout = oracle_repair_hankel_rollout(
        program,
        initial_history=initial,
        exact_control_tokens=controls,
        exact_target_tokens=torch.tensor(targets),
        exact_hidden_states=torch.stack(exact_hidden),
    )
    assert rollout.statistics.repairs == 0
    assert rollout.statistics.accepted_predictions == steps
    assert rollout.statistics.emitted_exact_rate == 1.0
    assert rollout.statistics.mean_repair_interval == steps


def test_one_wrong_target_charges_one_exact_repair() -> None:
    program, hidden, tokens = _program_and_trajectory()
    start = 31
    steps = 8
    initial = [reduce_hidden(program, hidden[start])]
    controls = tokens[start + 1 : start + 1 + steps]

    from vortex_runtime.hankel_decision_program import (
        decision_logits,
        predict_next_reduced,
        reconstruct_hidden,
    )

    history = list(initial)
    targets = []
    exact_hidden = []
    for control in controls:
        next_state = predict_next_reduced(program, history, int(control.item()))
        targets.append(int(torch.argmax(decision_logits(program, next_state)).item()))
        exact_hidden.append(reconstruct_hidden(program, next_state))
        history = [next_state]
    targets[3] = (targets[3] + 1) % program.control_table.shape[0]

    rollout = oracle_repair_hankel_rollout(
        program,
        initial_history=initial,
        exact_control_tokens=controls,
        exact_target_tokens=torch.tensor(targets),
        exact_hidden_states=torch.stack(exact_hidden),
    )
    assert rollout.statistics.repairs == 1
    assert rollout.statistics.repair_positions == (4,)
    assert rollout.statistics.mismatch_repairs == 1
    assert rollout.statistics.projected_repair_traffic_gib_per_token == (
        TARGET_REPAIR_TRAFFIC_GIB / steps
    )
    assert rollout.statistics.projected_repair_compute_gflop_per_token == (
        TARGET_REPAIR_COMPUTE_GFLOP / steps
    )


def test_256_token_single_repair_passes_strong_interval_but_compute_is_accounted() -> None:
    from vortex_runtime.oracle_hankel_repair import OracleRepairStatistics

    statistics = OracleRepairStatistics(
        tokens=256,
        repairs=1,
        accepted_predictions=255,
        accepted_fraction=255 / 256,
        mean_repair_interval=256.0,
        minimum_repair_interval=1,
        maximum_repair_interval=255,
        p50_repair_interval=128.0,
        p90_repair_interval=229.4,
        repair_positions=(1,),
        nonfinite_repairs=0,
        mismatch_repairs=1,
        emitted_exact_rate=1.0,
        projected_repair_traffic_gib_per_token=TARGET_REPAIR_TRAFFIC_GIB / 256,
        projected_repair_compute_gflop_per_token=TARGET_REPAIR_COMPUTE_GFLOP / 256,
    )
    budget = hankel_program_budget(
        state_rank=32,
        control_rank=16,
        order=2,
        lift="full",
    )
    passes, compute = repair_envelope_passes(
        statistics,
        program_hot_compute_gflop_per_token=budget.hot_compute_gflop_per_token,
        program_build_compute_gflop=budget.build_compute_gflop,
        horizon_tokens=256,
    )
    assert passes
    assert compute < 9.6
