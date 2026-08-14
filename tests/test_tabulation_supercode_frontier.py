from __future__ import annotations

import unittest

from vortex_runtime.tabulation_supercode_frontier import (
    DECISION,
    anf_coefficients,
    block_capacity_case,
    derive_audit,
    distinct_binary_rank_one_masks,
    first_capacity_case_meeting_target,
    linear_truth_table,
    selection_ball,
    verify_and_linearize_nonadaptive_xor,
)


class TabulationSupercodeFrontierTests(unittest.TestCase):
    def test_anf_transform_identifies_linear_and_quadratic_terms(self) -> None:
        table = tuple(
            (((point >> 0) & 1) ^ (((point >> 1) & 1) & ((point >> 2) & 1)))
            for point in range(8)
        )
        coefficients = anf_coefficients(table, 3)
        self.assertEqual(coefficients[1], 1)
        self.assertEqual(coefficients[6], 1)
        self.assertEqual(sum(coefficients), 2)

    def test_nonlinear_cells_linearize_with_same_recovery_sets(self) -> None:
        tables = []
        for cell in range(2):
            values = []
            for point in range(8):
                x0 = point & 1
                x1 = (point >> 1) & 1
                x2 = (point >> 2) & 1
                values.append((x0 ^ (x1 & x2)) if cell == 0 else (x1 & x2))
            tables.append(tuple(values))
        result = verify_and_linearize_nonadaptive_xor(
            cell_truth_tables=tables,
            recovery_sets={1: (0, 1)},
            dimension=3,
        )
        self.assertEqual(result["nonlinear_input_cell_count"], 2)
        self.assertEqual(result["linearized_atom_masks"], [1, 0])
        self.assertTrue(result["all_queries_exact_after_linearization"])

    def test_invalid_recovery_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            verify_and_linearize_nonadaptive_xor(
                cell_truth_tables=[linear_truth_table(1, 2)],
                recovery_sets={2: (0,)},
                dimension=2,
            )

    def test_rank_one_count_and_selection_ball_are_exact(self) -> None:
        self.assertEqual(distinct_binary_rank_one_masks(3), 50)
        self.assertEqual(selection_ball(10, 2), 56)

    def test_first_capacity_witness_is_side_23_but_not_constructor(self) -> None:
        result = first_capacity_case_meeting_target()
        self.assertEqual(result["block_side"], 23)
        self.assertEqual(result["ambient_dimension"], 529)
        self.assertEqual(result["favorable_dictionary_cells"], 619)
        self.assertEqual(result["minimum_weight_not_rejected_by_capacity"], 6)
        self.assertEqual(result["probe_fraction"], "6/529")
        self.assertTrue(result["capacity_threshold_meets_registered_fraction"])
        self.assertFalse(result["capacity_is_a_construction"])
        self.assertFalse(result["rank_one_alignment_established"])

    def test_neighboring_capacity_case_is_nonmonotone(self) -> None:
        result = block_capacity_case(24)
        self.assertEqual(result["minimum_weight_not_rejected_by_capacity"], 7)
        self.assertEqual(result["probe_fraction"], "7/576")
        self.assertFalse(result["capacity_threshold_meets_registered_fraction"])

    def test_random_atoms_do_not_supply_alignment_witness(self) -> None:
        result = block_capacity_case(23)
        self.assertEqual(
            result["random_atoms_fixed_query_hit_probability_log2_floor_upper"],
            -483,
        )
        self.assertEqual(
            result["random_atoms_expected_nonzero_rank_one_hits_log2_floor_upper"],
            -437,
        )

    def test_audit_keeps_only_the_uncovered_models_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertTrue(boundary["arbitrary_boolean_preprocessing_within_scope"])
        self.assertFalse(boundary["adaptive_data_dependent_addresses_covered"])
        self.assertFalse(boundary["arbitrary_word_decoder_covered"])
        self.assertFalse(boundary["constructor_supplied"])
        self.assertIn("NO_SURVIVING_CANDIDATE", result["decisions"])


if __name__ == "__main__":
    unittest.main()
