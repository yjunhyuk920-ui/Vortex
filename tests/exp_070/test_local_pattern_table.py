import numpy as np
import pytest

from vortex_runtime.local_pattern_table import (
    LocalPatternError,
    analyze_local_pattern_plan,
    bit_reversal_order,
    choose_joint_plan,
    registered_orders,
)


def test_repeated_patterns_reconstruct_and_reduce():
    matrix = np.asarray([
        [1, 2, 3, 4], [1, 2, 5, 6], [1, 2, 3, 4], [1, 2, 5, 6],
    ], dtype=np.int8)
    plan = analyze_local_pattern_plan(matrix, block_width=2, order_name="natural", order=range(4))
    assert plan.reconstruction_mismatches == 0
    assert plan.blocks[0].distinct_pattern_count == 1
    assert plan.blocks[1].distinct_pattern_count == 2
    assert plan.dictionary_bits < plan.dense_q4_bits


def test_forced_unique_patterns_do_not_look_free():
    matrix = np.asarray([[row, row + 1, row + 2, row + 3] for row in range(8)], dtype=np.int8)
    plan = analyze_local_pattern_plan(matrix, block_width=4, order_name="natural", order=range(4))
    assert plan.blocks[0].distinct_pattern_count == 8
    assert plan.operation_fraction > 1.0


def test_one_nibble_mutation_splits_dictionary_class():
    matrix = np.asarray([[1, 2], [1, 2], [1, 2]], dtype=np.int8)
    base = analyze_local_pattern_plan(matrix, block_width=2, order_name="natural", order=range(2))
    matrix[1, 1] = 3
    changed = analyze_local_pattern_plan(matrix, block_width=2, order_name="natural", order=range(2))
    assert changed.distinct_pattern_total == base.distinct_pattern_total + 1


def test_registered_orders_are_permutations_and_reconstruct():
    matrix = np.arange(64, dtype=np.int8).reshape(8, 8)
    for name, order in registered_orders(matrix, ("natural", "bit_reversal", "lexicographic_signature")):
        plan = analyze_local_pattern_plan(matrix, block_width=3, order_name=name, order=order)
        assert plan.reconstruction_mismatches == 0
        assert sorted(order) == list(range(8))


def test_bit_reversal_rejects_non_power_of_two():
    with pytest.raises(LocalPatternError):
        bit_reversal_order(6)


def test_joint_choice_does_not_cherry_pick_cost_axes():
    matrix = np.asarray([[1, 2, 1, 2], [1, 2, 3, 4], [1, 2, 1, 2]], dtype=np.int8)
    plans = [
        analyze_local_pattern_plan(matrix, block_width=width, order_name="natural", order=range(4))
        for width in (2, 4)
    ]
    selected = choose_joint_plan(plans)
    assert selected.joint_fraction == min(plan.joint_fraction for plan in plans)


def test_random_q4_has_high_pattern_entropy():
    rng = np.random.default_rng(700070)
    matrix = rng.integers(-7, 8, size=(128, 16), dtype=np.int8)
    plan = analyze_local_pattern_plan(matrix, block_width=8, order_name="natural", order=range(16))
    assert all(block.distinct_pattern_count >= 120 for block in plan.blocks)
    assert plan.operation_fraction > 0.9
