from __future__ import annotations

from fractions import Fraction
import unittest

from vortex_runtime.adaptive_packed_linear_word_gate import (
    DECISION,
    best_proportional_case,
    derive_audit,
    minimum_packed_word_probes,
    packed_word_support_capacity,
    proportional_packed_word_case,
)


class AdaptivePackedLinearWordGateTests(unittest.TestCase):
    def test_support_union_uses_exact_small_segre_intersections(self) -> None:
        # A one-dimensional span contains at most one nonzero rank-one mask.
        self.assertEqual(
            packed_word_support_capacity(
                rows=2, columns=2, words=4, word_bits=1, probes=1
            ),
            4,
        )
        probes, cases = minimum_packed_word_probes(
            rows=2, columns=2, words=4, word_bits=1
        )
        self.assertEqual(probes, 2)
        self.assertTrue(cases[0]["rejected"])
        self.assertFalse(cases[1]["rejected"])

    def test_registered_frontier_needs_at_least_eight_words(self) -> None:
        case = proportional_packed_word_case(rows=31, columns=42)
        self.assertEqual(case["stored_linear_bits"], 1_523)
        self.assertEqual(case["padded_physical_words"], 24)
        self.assertEqual(case["minimum_probes_not_count_rejected"], 8)
        self.assertTrue(case["probe_cases"][-2]["rejected"])
        self.assertEqual(
            Fraction(case["favorable_traffic_fraction"]), Fraction(64, 651)
        )
        self.assertGreater(
            Fraction(case["minimum_to_target_multiplier"]), 8
        )

    def test_every_registered_rectangle_misses_the_traffic_target(self) -> None:
        scan = best_proportional_case(maximum_side=128)
        self.assertEqual(scan["rectangles_checked"], 8_256)
        self.assertEqual(scan["registered_target_passing_count"], 0)
        best = scan["best_case"]
        self.assertEqual((best["rows"], best["columns"]), (118, 128))
        self.assertEqual(best["minimum_probes_not_count_rejected"], 22)
        self.assertEqual(
            Fraction(best["favorable_traffic_fraction"]), Fraction(11, 472)
        )

    def test_audit_does_not_overclaim_nonlinear_or_global_words(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertTrue(boundary["adaptive_addresses_covered"])
        self.assertTrue(boundary["arbitrary_exact_postprocessing_covered"])
        self.assertFalse(boundary["nonlinear_stored_words_covered"])
        self.assertFalse(boundary["global_cross_matrix_word_mixing_covered"])
        self.assertFalse(boundary["surviving_runtime_candidate"])


if __name__ == "__main__":
    unittest.main()
