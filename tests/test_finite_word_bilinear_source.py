from __future__ import annotations

from fractions import Fraction
import unittest

from vortex_runtime.finite_word_bilinear_source import (
    P50_TARGET_FRACTION,
    REQUESTED_UNIVERSAL_FRACTION,
    REJECT_DECISION,
    compile_block_truth_table,
    derive_audit,
    derive_block_truth_table_plan,
    derive_larsen_williams_boundary,
    derive_mailman_floor,
    gf2_bilinear,
    minimum_block_dimensions,
    query_block_truth_table,
    run_reference_controls,
)


class FiniteWordBilinearSourceTests(unittest.TestCase):
    def test_block_truth_table_matches_every_small_rank_one_query(self) -> None:
        matrix = (
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 0),
        )
        compiled = compile_block_truth_table(
            matrix,
            row_block_size=2,
            column_block_size=2,
        )
        for left_pattern in range(1 << 3):
            left = tuple(
                (left_pattern >> index) & 1 for index in range(3)
            )
            for right_pattern in range(1 << 3):
                right = tuple(
                    (right_pattern >> index) & 1
                    for index in range(3)
                )
                self.assertEqual(
                    query_block_truth_table(compiled, left, right),
                    gf2_bilinear(left, matrix, right),
                )

    def test_basis_queries_recover_every_matrix_bit(self) -> None:
        matrix = ((1, 0, 1), (1, 1, 0))
        compiled = compile_block_truth_table(
            matrix,
            row_block_size=1,
            column_block_size=2,
        )
        for row in range(2):
            left = tuple(int(index == row) for index in range(2))
            for column in range(3):
                right = tuple(int(index == column) for index in range(3))
                self.assertEqual(
                    query_block_truth_table(compiled, left, right),
                    matrix[row][column],
                )

    def test_reference_controls_are_exhaustive_and_deterministic(self) -> None:
        first = run_reference_controls(cases=12)
        second = run_reference_controls(cases=12)
        self.assertEqual(first, second)
        self.assertTrue(first["all_controls_pass"])
        self.assertEqual(
            first["exact_matches"], first["exhaustive_queries"]
        )

    def test_natural_word_plan_meets_query_axis_only_by_exploding_state(
        self,
    ) -> None:
        rows, columns, required_area = minimum_block_dimensions(word_bits=64)
        self.assertEqual((rows, columns, required_area), (37, 37, 1350))
        plan = derive_block_truth_table_plan(word_bits=64)
        self.assertEqual(plan["area"], 1369)
        self.assertEqual(plan["perimeter"], 74)
        self.assertEqual(
            plan["physical_word_traffic_fraction"], "16/1369"
        )
        self.assertEqual(
            plan["table_q4_storage_fraction"],
            "4722366482869645213696/1369",
        )
        self.assertTrue(plan["passes_query_operations"])
        self.assertTrue(plan["passes_physical_word_traffic"])
        self.assertFalse(plan["passes_one_checkpoint_storage_grant"])
        self.assertFalse(plan["passes_build_amortization"])
        self.assertFalse(plan["word_can_address_table"])
        self.assertEqual(plan["minimum_address_bits"], 100)
        self.assertGreater(
            Fraction(plan["amortized_build_fraction"]),
            P50_TARGET_FRACTION,
        )

    def test_requested_two_point_five_percent_still_explodes_state(
        self,
    ) -> None:
        plan = derive_block_truth_table_plan(
            word_bits=64,
            target_fraction=REQUESTED_UNIVERSAL_FRACTION,
        )
        self.assertEqual((plan["rows"], plan["columns"]), (25, 26))
        self.assertEqual(plan["area"], 650)
        self.assertEqual(plan["perimeter"], 51)
        self.assertEqual(plan["physical_word_traffic_fraction"], "8/325")
        self.assertEqual(
            plan["table_q4_storage_fraction"],
            "281474976710656/325",
        )
        self.assertEqual(
            plan["amortized_build_fraction"],
            "4398046511104/25390625",
        )
        self.assertEqual(plan["minimum_address_bits"], 78)
        self.assertTrue(plan["passes_query_operations"])
        self.assertTrue(plan["passes_physical_word_traffic"])
        self.assertFalse(plan["passes_one_checkpoint_storage_grant"])
        self.assertFalse(plan["passes_build_amortization"])
        self.assertFalse(plan["word_can_address_table"])
        self.assertGreater(
            Fraction(plan["table_q4_storage_fraction"]),
            800_000_000_000,
        )

    def test_thirty_two_bit_plan_cannot_hide_address_and_build_costs(
        self,
    ) -> None:
        plan = derive_block_truth_table_plan(word_bits=32)
        self.assertEqual((plan["rows"], plan["columns"]), (26, 26))
        self.assertEqual(plan["area"], 676)
        self.assertEqual(plan["minimum_address_bits"], 79)
        self.assertTrue(plan["passes_physical_word_traffic"])
        self.assertFalse(plan["word_can_address_table"])
        self.assertFalse(plan["passes_build_amortization"])

    def test_mailman_and_boolean_probe_routes_do_not_close_numerical_target(
        self,
    ) -> None:
        mailman = derive_mailman_floor()
        self.assertEqual(mailman["maximum_favorable_chunk_symbols"], 9)
        self.assertEqual(mailman["operation_fraction_floor"], "1/9")
        self.assertFalse(mailman["passes_target"])
        probe = derive_larsen_williams_boundary()
        self.assertEqual(
            probe["favorable_q4_relative_fraction"], "1/64"
        )
        self.assertEqual(probe["favorable_leading_one_probes"], 262_144)
        self.assertEqual(probe["favorable_redundant_bits"], 16_777_216)
        self.assertEqual(
            probe["theorem_output"], "BOOLEAN_RECTANGLE_NONEMPTY"
        )
        self.assertEqual(
            probe["exact_parity_count_or_signed_sum"], "NOT SUPPLIED"
        )

    def test_audit_keeps_the_general_cell_probe_question_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], REJECT_DECISION)
        boundary = result["claim_boundary"]
        self.assertEqual(
            boundary["general_nonlinear_rank_one_cell_probe_structure"],
            "OPEN",
        )
        self.assertEqual(
            boundary["finite_word_universal_impossibility"], "NOT PROVED"
        )
        self.assertEqual(
            boundary["universal_2_5_percent_guarantee"],
            "NOT ESTABLISHED",
        )
        self.assertEqual(boundary["core_candidate"], "NONE")

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            compile_block_truth_table(
                (),
                row_block_size=1,
                column_block_size=1,
            )
        with self.assertRaises(ValueError):
            compile_block_truth_table(
                ((1,),),
                row_block_size=0,
                column_block_size=1,
            )
        with self.assertRaises(ValueError):
            gf2_bilinear((1,), ((2,),), (1,))
        with self.assertRaises(ValueError):
            minimum_block_dimensions(word_bits=0)
        with self.assertRaises(ValueError):
            run_reference_controls(cases=0)


if __name__ == "__main__":
    unittest.main()
