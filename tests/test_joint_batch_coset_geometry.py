from __future__ import annotations

import unittest

from vortex_runtime.joint_batch_coset_geometry import (
    DECISION,
    derive_audit,
    factor_envelope_catalog_screen,
    joint_factor_envelope,
    maximum_rank_one_points_control,
    product_simplex_generalized_weight,
    registered_square_joint_batch_counting_screen,
    square_two_strip_maximum_rank_one_points,
)


class JointBatchCosetGeometryTests(unittest.TestCase):
    def test_product_simplex_control_sequences_match_exhaustive_prototype(self) -> None:
        expected = {
            (2, 2): [0, 1, 3, 5, 9],
            (2, 3): [0, 1, 3, 7, 9, 13, 21],
            (2, 4): [0, 1, 3, 7, 15, 17, 21, 29, 45],
        }
        for shape, sequence in expected.items():
            rows, columns = shape
            actual = [
                maximum_rank_one_points_control(rows, columns, dimension)
                for dimension in range(rows * columns + 1)
            ]
            self.assertEqual(actual, sequence)

    def test_two_strip_formula_agrees_with_generalized_weights_on_controls(self) -> None:
        for side in range(2, 7):
            for dimension in range(2 * side + 1):
                self.assertEqual(
                    square_two_strip_maximum_rank_one_points(side, dimension),
                    maximum_rank_one_points_control(side, side, dimension),
                )

    def test_batch_factor_envelope_is_only_1024_dimensional(self) -> None:
        result = joint_factor_envelope(16_384, 16_384, 32)
        self.assertEqual(result["left_factor_span_at_most"], 32)
        self.assertEqual(result["right_factor_span_at_most"], 32)
        self.assertEqual(result["tensor_envelope_dimension_at_most"], 1_024)

    def test_literal_factor_catalog_fails_before_native_width(self) -> None:
        result = factor_envelope_catalog_screen()
        self.assertEqual(
            result["factor_subspace_pair_count_log2_strict_lower"],
            1_046_528,
        )
        self.assertFalse(result["literal_catalog_fits"])
        self.assertFalse(result["implicit_shared_generator_rejected"])

    def test_general_counting_lower_bound_is_finite_but_too_small(self) -> None:
        result = registered_square_joint_batch_counting_screen()
        self.assertEqual(result["largest_bit_atom_support_ruled_out"], 20_217)
        self.assertEqual(result["minimum_bit_atoms_not_ruled_out"], 20_218)
        self.assertEqual(
            result["minimum_perfectly_packed_64bit_words_not_ruled_out"],
            316,
        )
        self.assertFalse(result["is_target_scale_lower_bound"])

    def test_audit_keeps_only_implicit_generator_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertTrue(boundary["rank_one_intersection_formula_exact"])
        self.assertFalse(boundary["implicit_joint_envelope_generator_constructed"])
        self.assertFalse(boundary["native_numerical_lift"])
        self.assertFalse(boundary["target_candidate"])
        self.assertIn("NO_SURVIVING_CANDIDATE", result["decisions"])

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            product_simplex_generalized_weight(0, 2, 1)
        with self.assertRaises(ValueError):
            square_two_strip_maximum_rank_one_points(4, 9)
        with self.assertRaises(ValueError):
            joint_factor_envelope(0, 2, 1)
        with self.assertRaises(ValueError):
            factor_envelope_catalog_screen(4, 4)


if __name__ == "__main__":
    unittest.main()
