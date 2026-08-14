from __future__ import annotations

import unittest

from vortex_runtime.nonlinear_recursive_lift_gate import (
    affine_coefficients,
    derive_audit,
    exhaustive_arity_audit,
    two_site_factorization_witness,
    two_site_parity_factors,
)


class NonlinearRecursiveLiftGateTests(unittest.TestCase):
    def test_affine_truth_tables_factor_through_child_parities(self) -> None:
        # f(x0,x1,x2)=1 XOR x0 XOR x2.
        truth_table = 0
        for point in range(8):
            value = 1 ^ (point & 1) ^ ((point >> 2) & 1)
            truth_table |= value << point
        self.assertEqual(
            affine_coefficients(arity=3, truth_table=truth_table),
            (1, (1, 0, 1)),
        )
        self.assertTrue(
            two_site_parity_factors(arity=3, truth_table=truth_table)
        )

    def test_and_has_an_explicit_same_child_different_parent_witness(self) -> None:
        witness = two_site_factorization_witness(
            arity=2, truth_table=1 << 3
        )
        self.assertIsNotNone(witness)
        assert witness is not None
        self.assertEqual(witness["child_parity_vector"], 3)
        self.assertNotEqual(
            witness["first_parent_parity"],
            witness["second_parent_parity"],
        )

    def test_exhaustive_small_functions_match_affinity_exactly(self) -> None:
        rows = exhaustive_arity_audit(maximum_arity=4)
        self.assertEqual(
            [row["boolean_function_count"] for row in rows],
            [4, 16, 256, 65_536],
        )
        self.assertEqual(
            [row["affine_function_count"] for row in rows],
            [4, 8, 16, 32],
        )
        self.assertTrue(
            all(row["classification_mismatch_count"] == 0 for row in rows)
        )

    def test_audit_keeps_global_correlation_encodings_open(self) -> None:
        audit = derive_audit()
        self.assertFalse(
            audit["consequence"]["free_recursive_lift_of_nonlinear_seed"]
        )
        self.assertTrue(
            audit["consequence"]["additional_correlation_sources_required"]
        )
        boundary = audit["claim_boundary"]
        self.assertTrue(boundary["entrywise_black_box_recursion_covered"])
        self.assertFalse(boundary["non_entrywise_global_encodings_covered"])
        self.assertFalse(boundary["surviving_runtime_candidate"])

    def test_invalid_truth_tables_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            affine_coefficients(arity=0, truth_table=0)
        with self.assertRaises(ValueError):
            two_site_parity_factors(arity=2, truth_table=16)


if __name__ == "__main__":
    unittest.main()
