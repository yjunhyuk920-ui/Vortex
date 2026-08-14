from __future__ import annotations

from fractions import Fraction
import unittest

from vortex_runtime.adaptive_codebook_trapdoor_frontier import (
    DECISION,
    derive_audit,
    hamming_ball_volume,
    nearest_pair_literal_table_gate,
    nearest_pair_scalar_reference,
    trapshift_reference,
)


class AdaptiveCodebookTrapdoorFrontierTests(unittest.TestCase):
    def test_hamming_ball_volume_small_exact(self) -> None:
        self.assertEqual(hamming_ball_volume(5, 0), 1)
        self.assertEqual(hamming_ball_volume(5, 2), 16)
        self.assertEqual(hamming_ball_volume(5, 5), 32)

    def test_small_pair_gate_matches_exact_packing_formula(self) -> None:
        dimension = 8
        target = Fraction(1, 16)
        result = nearest_pair_literal_table_gate(
            dimension=dimension,
            target_fraction=target,
            hot_bits=1 << 20,
        )
        budget = int(target * dimension * dimension)
        removal_radius = (2 * budget) // dimension
        expected_entries = -(
            -(1 << dimension)
            // hamming_ball_volume(dimension, removal_radius)
        )
        self.assertEqual(result.packing_removal_radius, removal_radius)
        self.assertEqual(result.minimum_literal_table_entries, expected_entries)

    def test_registered_literal_pair_table_is_decisively_too_large(self) -> None:
        result = nearest_pair_literal_table_gate()
        self.assertEqual(result.dimension, 16_384)
        self.assertGreater(result.exponent_deficit_bits, 10_000.0)
        self.assertGreater(result.literal_table_address_bits, 64)
        self.assertFalse(result.fits_hot_capacity)

    def test_nearest_pair_reference_matches_dense_gf2(self) -> None:
        matrix = [
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 1],
        ]
        left = [1, 1, 0]
        right = [0, 1, 1, 1]
        left_codeword = [1, 0, 0]
        right_codeword = [0, 1, 0, 1]
        observed, reads = nearest_pair_scalar_reference(
            matrix,
            left,
            right,
            left_codeword,
            right_codeword,
        )
        expected = 0
        for row_index, left_value in enumerate(left):
            for column_index, right_value in enumerate(right):
                expected ^= (
                    left_value
                    & matrix[row_index][column_index]
                    & right_value
                )
        self.assertEqual(observed, expected)
        expected_reads = sum(
            (left[row] & right[column])
            != (left_codeword[row] & right_codeword[column])
            for row in range(3)
            for column in range(4)
        )
        self.assertEqual(reads, expected_reads)

    def test_nearest_pair_reference_handles_removed_cached_row(self) -> None:
        observed, reads = nearest_pair_scalar_reference(
            [[1, 1], [1, 0]],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
        )
        self.assertEqual(observed, 1)
        self.assertEqual(reads, 3)

    def test_trapshift_identity_is_exact_over_small_field(self) -> None:
        matrix = [[1, 4, 2], [3, 0, 6]]
        mask = [[5, 1, 3], [2, 4, 1]]
        vector = [6, 2, 5]
        recovered, shifted, mask_product = trapshift_reference(
            matrix, mask, vector, 7
        )
        expected = [
            sum(cell * value for cell, value in zip(row, vector)) % 7
            for row in matrix
        ]
        self.assertEqual(recovered, expected)
        self.assertEqual(
            recovered,
            [(a - b) % 7 for a, b in zip(shifted, mask_product)],
        )

    def test_audit_keeps_general_gap_open(self) -> None:
        audit = derive_audit()
        self.assertEqual(audit["decision"], DECISION)
        self.assertFalse(audit["old_2_5_percent_assumption_used"])
        self.assertFalse(
            audit["claim_boundary"]["general_compressed_bilinear_oracle_rejected"]
        )
        self.assertFalse(audit["claim_boundary"]["target_achieved"])

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            hamming_ball_volume(0, 0)
        with self.assertRaises(ValueError):
            nearest_pair_literal_table_gate(target_fraction=Fraction(1))
        with self.assertRaises(ValueError):
            trapshift_reference([[1]], [[1, 2]], [1], 7)
        with self.assertRaises(ValueError):
            nearest_pair_scalar_reference([[1]], [2], [1], [0], [0])


if __name__ == "__main__":
    unittest.main()
