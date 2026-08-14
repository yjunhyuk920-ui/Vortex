from __future__ import annotations

from fractions import Fraction
import itertools
import unittest

from vortex_runtime.linear_functional_decoder_activity import (
    BATCH_QUERIES,
    DECISION,
    HOT_BITS,
    REGISTERED_NON_EMBEDDING_BITS,
    batch_union_activity_lower,
    bilinear_one_probability,
    derive_audit,
    fixed_linear_decoder_cold_floor,
)


def _bilinear(matrix: list[list[int]], left: int, right: int) -> int:
    rows = len(matrix)
    columns = len(matrix[0])
    return sum(
        ((left >> i) & 1) * matrix[i][j] * ((right >> j) & 1)
        for i in range(rows)
        for j in range(columns)
    ) % 2


def _gf2_rank(matrix: list[list[int]]) -> int:
    rows = [
        sum(value << column for column, value in enumerate(row))
        for row in matrix
    ]
    basis: dict[int, int] = {}
    for row in rows:
        while row:
            pivot = row.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = row
                break
            row ^= basis[pivot]
    return len(basis)


class LinearFunctionalDecoderActivityTests(unittest.TestCase):
    def test_bilinear_activity_formula_matches_exhaustive_small_matrices(
        self,
    ) -> None:
        for rows, columns in ((2, 2), (2, 3), (3, 2)):
            for bits in range(1 << (rows * columns)):
                matrix = [
                    [
                        (bits >> (i * columns + j)) & 1
                        for j in range(columns)
                    ]
                    for i in range(rows)
                ]
                rank = _gf2_rank(matrix)
                ones = sum(
                    _bilinear(matrix, left, right)
                    for left in range(1 << rows)
                    for right in range(1 << columns)
                )
                probability = Fraction(ones, 1 << (rows + columns))
                self.assertEqual(probability, bilinear_one_probability(rank))

    def test_independent_global_blocks_multiply_character_biases(self) -> None:
        # Two nonzero rank-one forms, one in each independent 2x2 block.
        matrix = [[1, 0], [0, 0]]
        ones = 0
        total = 0
        for left1, right1, left2, right2 in itertools.product(
            range(4), repeat=4
        ):
            value = _bilinear(matrix, left1, right1) ^ _bilinear(
                matrix, left2, right2
            )
            ones += value
            total += 1
        self.assertEqual(Fraction(ones, total), Fraction(3, 8))
        self.assertGreaterEqual(Fraction(ones, total), Fraction(1, 4))

    def test_batch_union_probability_is_nearly_one(self) -> None:
        value = batch_union_activity_lower(BATCH_QUERIES)
        self.assertEqual(
            value, Fraction(18_444_891_053_520_699_775, 1 << 64)
        )
        self.assertGreater(float(value), 0.99989)

    def test_registered_floor_grants_all_hot_bits_and_word_packing(self) -> None:
        result = fixed_linear_decoder_cold_floor()
        self.assertEqual(
            result["data_dimension_bits"], REGISTERED_NON_EMBEDDING_BITS
        )
        self.assertEqual(result["hot_one_bit_cells_granted"], HOT_BITS)
        self.assertEqual(
            result["minimum_cold_nonzero_decoder_rows"], 335_028_420_608
        )
        self.assertEqual(
            result["forced_worst_case_cold_active_cell_bits"],
            334_994_766_191,
        )
        self.assertEqual(
            result["forced_worst_case_cold_words"], 5_234_293_222
        )
        self.assertEqual(
            result["forced_worst_case_cold_bytes"], 41_874_345_776
        )

    def test_audit_rejects_only_fixed_linear_decoder_scope(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        io = result["io_comparison"]
        self.assertGreater(io["cold_over_requested_block"], 3.03)
        self.assertGreater(io["zero_compute_ms_per_token"], 40.8)
        self.assertTrue(io["exceeds_20ms_per_token_before_other_costs"])
        boundary = result["claim_boundary"]
        self.assertEqual(boundary["fixed_linear_decoder"], "REJECTED")
        self.assertEqual(boundary["invertible_basis_change"], "COVERED")
        self.assertEqual(
            boundary["nonlinear_minimum_weight_syndrome_decoder"],
            "NOT_COVERED",
        )
        self.assertEqual(
            boundary["causal_transformer_reachability_of_independent_batch"],
            "NOT_ESTABLISHED",
        )
        self.assertFalse(boundary["target_candidate"])

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            bilinear_one_probability(-1)
        with self.assertRaises(ValueError):
            batch_union_activity_lower(0)
        with self.assertRaises(ValueError):
            fixed_linear_decoder_cold_floor(data_bits=10, hot_bits=10)


if __name__ == "__main__":
    unittest.main()

