from __future__ import annotations

import random
import unittest

from vortex_runtime.implicit_direct_query_gate import (
    compile_pattern_image_frame,
    compile_value_class_index,
    direct_gf2_matvec_mask,
    direct_ordered_finite_word_matvec,
    exhaustive_hadamard_automorphism_control,
    exhaustive_boolean_meet_automorphism_control,
    exhaustive_linear_hadamard_isotopy_control,
    fp32_exact_integer_state_coalescing_adversary,
    gauge_dense_support_invariance,
    minimum_equal_block_width_for_target,
    pattern_image_storage_atoms,
    query_pattern_image_frame,
    query_value_class_index,
    query_value_state_coalesced,
    registered_sparse_image_frame_audit,
    registered_one_sided_frame_lower_bounds,
    ten_eleven_partition,
    value_class_operation_bounds,
)


class ImplicitDirectQueryGateTests(unittest.TestCase):
    def test_registered_partition_hits_ten_percent_binary_route(self):
        for width in (1024, 16384, 53248, 128256):
            blocks = ten_eleven_partition(width)
            self.assertEqual(sum(blocks), width)
            self.assertLessEqual(len(blocks) / width, 0.1)
            self.assertTrue(all(block in (10, 11) for block in blocks))

    def test_pattern_image_frame_is_exact_for_every_small_query(self):
        rng = random.Random(26090901)
        matrix = [[rng.randrange(2) for _ in range(7)] for _ in range(5)]
        frame = compile_pattern_image_frame(matrix, (3, 4))
        for vector in range(1 << 7):
            got, _ = query_pattern_image_frame(frame, vector)
            self.assertEqual(got, direct_gf2_matvec_mask(matrix, vector))

    def test_pattern_image_frame_addresses_need_no_checkpoint_program(self):
        matrix = [[(i + j) & 1 for j in range(6)] for i in range(4)]
        frame = compile_pattern_image_frame(matrix, (3, 3))
        vector = 0b101011
        _, reads = query_pattern_image_frame(frame, vector)
        self.assertEqual(reads, 2)
        self.assertEqual(sum(len(images) for images in frame.images), 14)

    def test_registered_sparse_image_frame_storage_is_fatal(self):
        audit = registered_sparse_image_frame_audit()
        self.assertTrue(audit["removes_at_least_90_percent_binary_source"])
        self.assertGreater(audit["storage_over_binary_source"], 100.0)
        self.assertGreater(audit["target_multiple"], 8.0)
        self.assertFalse(audit["native_universal_fast_path_proved"])

    def test_target_equal_block_tradeoff_starts_at_85(self):
        self.assertEqual(minimum_equal_block_width_for_target(), 85)
        self.assertGreater(pattern_image_storage_atoms((85,)), 1 << 84)

    def test_general_complete_image_frame_still_needs_large_redundancy(self):
        audit = registered_one_sided_frame_lower_bounds()
        for row in audit["rows"]:
            self.assertGreater(row["minimum_complete_image_storage_ratio_90"], 30.0)
            self.assertGreater(row["minimum_complete_image_storage_ratio_target"], 1e20)

    def test_value_class_executor_is_exact_in_declared_finite_word_abi(self):
        rng = random.Random(26090902)
        matrix = [[rng.randrange(8) for _ in range(9)] for _ in range(7)]
        vector = [rng.randrange(16) for _ in range(9)]
        mul = lambda a, b: (a * b) & 0xFF
        add = lambda a, b: (a + b) & 0xFF
        index = compile_value_class_index(matrix)
        got, stats = query_value_class_index(index, vector, multiply=mul, add=add, zero=0)
        expected = direct_ordered_finite_word_matvec(matrix, vector, multiply=mul, add=add, zero=0)
        self.assertEqual(got, expected)
        self.assertEqual(stats["memberships"], 7 * 9)

    def test_value_class_best_case_still_fails_90_percent_operation_gate(self):
        best = value_class_operation_bounds(16384, 16384, 1)
        self.assertGreater(best["candidate_operation_fraction_decimal"], 0.49)
        self.assertFalse(best["removes_at_least_90_percent_operations"])

    def test_value_class_high_distinctness_restores_full_products(self):
        worst = value_class_operation_bounds(1024, 16384, 1024)
        self.assertEqual(worst["candidate_multiplies"], worst["baseline_multiplies"])
        self.assertFalse(worst["removes_at_least_90_percent_operations"])

    def test_state_dependent_value_coalescing_is_exact(self):
        matrix = [
            [1, 1, 1, 1],
            [2, 1, 1, 1],
            [3, 1, 1, 1],
            [4, 1, 1, 1],
        ]
        vector = [1, 1, 1, 1]
        mul = lambda a, b: a * b
        add = lambda a, b: a + b
        got, stats = query_value_state_coalesced(matrix, vector, multiply=mul, add=add, zero=0)
        expected = direct_ordered_finite_word_matvec(matrix, vector, multiply=mul, add=add, zero=0)
        self.assertEqual(got, expected)
        self.assertEqual(stats["state_weight_pair_updates"], 16)

    def test_fp32_exact_integer_adversary_defeats_state_coalescing(self):
        result = fp32_exact_integer_state_coalescing_adversary(16_384, 16_384)
        self.assertTrue(result["fp32_integer_operations_exact"])
        self.assertEqual(result["state_coalescing_update_fraction"], 1.0)
        self.assertFalse(result["removes_at_least_90_percent_state_updates"])

    def test_hadamard_linear_automorphisms_are_only_permutations_on_controls(self):
        expected = {1: (1, 1), 2: (6, 2), 3: (168, 6)}
        for dimension, (invertible, preserving) in expected.items():
            result = exhaustive_hadamard_automorphism_control(dimension)
            self.assertEqual(result["invertible_maps"], invertible)
            self.assertEqual(result["product_preserving_maps"], preserving)
            self.assertEqual(result["product_preserving_nonpermutations"], 0)

    def test_even_arbitrary_meet_bijections_are_coordinate_permutations_on_controls(self):
        expected = {1: (2, 1), 2: (24, 2), 3: (40320, 6)}
        for dimension, (checked, preserving) in expected.items():
            result = exhaustive_boolean_meet_automorphism_control(dimension)
            self.assertEqual(result["bijections_checked"], checked)
            self.assertEqual(result["meet_preserving_bijections"], preserving)
            self.assertEqual(result["meet_preserving_non_coordinate_permutations"], 0)

    def test_independent_linear_gauges_preserving_product_are_aligned_permutations(self):
        expected = {1: (1, 1), 2: (216, 2)}
        for dimension, (triples, preserving) in expected.items():
            result = exhaustive_linear_hadamard_isotopy_control(dimension)
            self.assertEqual(result["triples_checked"], triples)
            self.assertEqual(result["product_isotopies"], preserving)
            self.assertEqual(result["non_aligned_permutation_isotopies"], 0)

    def test_safe_gauge_preserves_dense_support(self):
        result = gauge_dense_support_invariance(128, 256)
        self.assertEqual(result["dense_nonzeros"], result["nonzeros_after_any_row_column_permutation"])
        self.assertFalse(result["removes_at_least_90_percent"])


if __name__ == "__main__":
    unittest.main()
