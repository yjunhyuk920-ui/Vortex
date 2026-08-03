from __future__ import annotations

import numpy as np
import pytest

from vortex_runtime.fixed_point import (
    FixedPointError,
    GatedTriangularChain,
    MapEvaluation,
    matching_prefix,
    run_anderson,
    run_damped_picard,
    triangular_transcript_indistinguishable,
)


def linear_map(matrix: np.ndarray, bias: np.ndarray):
    def apply(state: np.ndarray) -> MapEvaluation:
        projected = state @ matrix.T + bias
        hard = tuple(int(value >= 0.0) for value in projected[:, 0])
        return MapEvaluation(projected, hard, projected.nbytes, projected.size)

    return apply


def test_picard_records_exact_accounting() -> None:
    initial = np.zeros((4, 2), dtype=np.float64)
    matrix = np.eye(2) * 0.5
    bias = np.ones((4, 2), dtype=np.float64)
    result = run_damped_picard(
        initial,
        map_fn=linear_map(matrix, bias),
        iterations=4,
        damping=1.0,
        record_steps=(1, 2, 4),
    )
    assert result.target_solver_full_streams == 4
    assert [snapshot.iteration for snapshot in result.snapshots] == [1, 2, 4]
    assert result.projection_read_bytes == 4 * initial.nbytes
    assert np.all(np.isfinite(result.final_state))
    assert result.snapshot_at(4).residual_l2 < result.snapshot_at(1).residual_l2


def test_anderson_improves_linear_contraction_positive_control() -> None:
    initial = np.zeros((8, 1), dtype=np.float64)
    matrix = np.asarray([[0.95]], dtype=np.float64)
    bias = np.ones((8, 1), dtype=np.float64) * 0.05
    mapping = linear_map(matrix, bias)
    picard = run_damped_picard(initial, map_fn=mapping, iterations=4, damping=1.0)
    anderson = run_anderson(
        initial,
        map_fn=mapping,
        iterations=4,
        history_size=4,
        damping=1.0,
        regularization=1e-10,
    )
    assert anderson.snapshot_at(4).residual_l2 < picard.snapshot_at(4).residual_l2
    assert anderson.target_solver_full_streams == picard.target_solver_full_streams == 4


def test_anderson_ill_conditioning_falls_back_without_nan() -> None:
    initial = np.zeros((4, 2), dtype=np.float64)

    def constant(state: np.ndarray) -> MapEvaluation:
        projected = np.ones_like(state)
        return MapEvaluation(projected, (1, 1, 1, 1))

    result = run_anderson(
        initial,
        map_fn=constant,
        iterations=4,
        history_size=4,
        damping=1.0,
        regularization=0.0,
        condition_limit=10.0,
    )
    assert result.numerical_fallbacks >= 1
    assert np.all(np.isfinite(result.final_state))


def test_map_nan_and_shape_faults_fail_closed() -> None:
    initial = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(FixedPointError):
        run_damped_picard(
            initial,
            map_fn=lambda state: MapEvaluation(np.zeros((1, 2)), (0,)),
            iterations=1,
            damping=1.0,
            record_steps=(1,),
        )
    with pytest.raises(FixedPointError):
        run_damped_picard(
            initial,
            map_fn=lambda state: MapEvaluation(np.full_like(state, np.nan), (0, 0)),
            iterations=1,
            damping=1.0,
            record_steps=(1,),
        )


def make_chain(length: int = 12) -> GatedTriangularChain:
    return GatedTriangularChain(
        tuple(range(1, length + 1)), vocab_size=length + 2, decoy_token=0
    )


def test_triangular_picard_reveals_at_most_one_new_position_per_round() -> None:
    chain = make_chain()
    result = run_damped_picard(
        chain.zero_state(),
        map_fn=chain.map,
        iterations=4,
        damping=1.0,
        record_steps=(1, 2, 3, 4),
    )
    for snapshot in result.snapshots:
        assert matching_prefix(snapshot.hard_tokens, chain.exact_tokens) == snapshot.iteration


def test_triangular_anderson_does_not_break_round_barrier() -> None:
    chain = make_chain()
    result = run_anderson(
        chain.zero_state(),
        map_fn=chain.map,
        iterations=4,
        history_size=4,
        damping=1.0,
        regularization=1e-8,
        record_steps=(1, 2, 3, 4),
    )
    for snapshot in result.snapshots:
        assert matching_prefix(snapshot.hard_tokens, chain.exact_tokens) <= snapshot.iteration
    assert matching_prefix(result.snapshot_at(4).hard_tokens, chain.exact_tokens) <= 4


def test_hidden_suffix_is_indistinguishable_before_predecessor_resolution() -> None:
    first = GatedTriangularChain((1, 2, 3, 4, 5, 6), vocab_size=10, decoy_token=0)
    second = GatedTriangularChain((1, 2, 7, 8, 9, 6), vocab_size=10, decoy_token=0)
    state = np.zeros((6, 10), dtype=np.float64)
    state[0, 1] = 1.0
    assert triangular_transcript_indistinguishable(
        first, second, state, unresolved_position=2
    )


def test_future_information_label_is_preserved() -> None:
    chain = make_chain(4)
    result = run_damped_picard(
        chain.zero_state(),
        map_fn=chain.map,
        iterations=1,
        damping=1.0,
        record_steps=(1,),
        future_information_used=True,
    )
    assert result.future_information_used


def test_invalid_solver_parameters_are_rejected() -> None:
    chain = make_chain(4)
    with pytest.raises(FixedPointError):
        run_anderson(
            chain.zero_state(),
            map_fn=chain.map,
            iterations=4,
            history_size=0,
            damping=1.0,
        )
    with pytest.raises(FixedPointError):
        run_damped_picard(
            chain.zero_state(),
            map_fn=chain.map,
            iterations=1,
            damping=0.0,
            record_steps=(1,),
        )
