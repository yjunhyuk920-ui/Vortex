from __future__ import annotations

from fractions import Fraction
import unittest

from vortex_runtime.nonlinear_bilinear_source import (
    BAUR_STRASSEN_UNIT_COST_FACTOR,
    P50_TARGET_FRACTION,
    REJECT_DECISION,
    build_obfuscated_bilinear_tape,
    derive_audit,
    exact_bilinear,
    exact_matvec,
    run_reference_controls,
)


class NonlinearBilinearSourceTests(unittest.TestCase):
    def test_exact_bilinear_and_matvec_reference(self) -> None:
        matrix = ((2, -3, 5), (7, 0, -1))
        right = (Fraction(1, 2), Fraction(-2, 3), Fraction(3, 5))
        left = (Fraction(4, 7), Fraction(-5, 2))
        product = exact_matvec(matrix, right)
        self.assertEqual(product, (Fraction(6), Fraction(29, 10)))
        self.assertEqual(
            exact_bilinear(left, matrix, right), Fraction(-107, 28)
        )

    def test_nonlinear_cancellation_paths_recover_exact_left_gradient(self) -> None:
        matrix = ((3, -2), (5, 7), (-1, 4))
        left = (Fraction(2, 3), Fraction(-3, 5), Fraction(7, 4))
        right = (Fraction(-5, 6), Fraction(9, 7))
        for branch in (0, 1):
            with self.subTest(branch=branch):
                tape, output, left_nodes = build_obfuscated_bilinear_tape(
                    left, matrix, right, branch=branch
                )
                self.assertEqual(
                    tape.nodes[output].value,
                    exact_bilinear(left, matrix, right),
                )
                self.assertEqual(
                    tape.gradient(output, left_nodes),
                    exact_matvec(matrix, right),
                )

    def test_reference_controls_cover_both_paths_and_exact_rationals(self) -> None:
        controls = run_reference_controls(cases=18)
        self.assertEqual(controls["branch_zero_cases"], 9)
        self.assertEqual(controls["branch_one_cases"], 9)
        self.assertEqual(controls["exact_value_matches"], 18)
        self.assertEqual(controls["exact_gradient_matches"], 18)
        self.assertTrue(controls["all_controls_pass"])

    def test_registered_containment_is_not_misstated_as_an_impossibility(
        self,
    ) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], REJECT_DECISION)
        containment = result["containment"]
        self.assertEqual(containment["baur_strassen_unit_cost_factor"], 4)
        self.assertEqual(
            containment["target_query_implies_static_matvec_fraction_at_most"],
            str(BAUR_STRASSEN_UNIT_COST_FACTOR * P50_TARGET_FRACTION),
        )
        self.assertEqual(
            containment["result_type"],
            "NOVELTY_CONTAINMENT_NOT_FINITE_IMPOSSIBILITY",
        )
        boundary = result["claim_boundary"]
        self.assertEqual(
            boundary["finite_word_bitwise_rounding_or_discontinuous_addressing"],
            "OPEN",
        )
        self.assertEqual(boundary["general_cell_probe_lower_bound"], "NOT PROVED")
        self.assertEqual(boundary["core_candidate"], "NONE")

    def test_invalid_reference_shapes_and_parameters_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            exact_matvec((), ())
        with self.assertRaises(ValueError):
            exact_matvec(((1, 2),), (1,))
        with self.assertRaises(ValueError):
            build_obfuscated_bilinear_tape((1,), ((1,),), (1,), branch=2)
        with self.assertRaises(ValueError):
            run_reference_controls(cases=0)


if __name__ == "__main__":
    unittest.main()
