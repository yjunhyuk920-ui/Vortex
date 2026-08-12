from __future__ import annotations

from fractions import Fraction
import unittest

from vortex_runtime.adaptive_nonlinear_probe_degree_gate import (
    DECISION,
    adaptive_nonlinear_probe_case,
    cylinder_polynomial_dimension,
    derive_audit,
    minimum_adaptive_nonlinear_probes,
    proportional_nonlinear_word_case,
    scan_proportional_nonlinear_words,
)
from vortex_runtime.segre_sparse_cover_fourier_gate import proportional_cells


class AdaptiveNonlinearProbeDegreeGateTests(unittest.TestCase):
    def test_cylinder_dimension_counts_nonconstant_alphabet_basis(self) -> None:
        self.assertEqual(
            cylinder_polynomial_dimension(cells=3, alphabet_bits=1, degree=1),
            4,
        )
        self.assertEqual(
            cylinder_polynomial_dimension(cells=2, alphabet_bits=2, degree=2),
            16,
        )

    def test_small_degree_gap_is_real_but_not_a_constructor(self) -> None:
        threshold = minimum_adaptive_nonlinear_probes(
            rows=2, columns=3, cells=7, alphabet_bits=1
        )
        self.assertEqual(threshold["minimum_probes_not_degree_rejected"], 2)
        one_probe = adaptive_nonlinear_probe_case(
            rows=2, columns=3, cells=7, alphabet_bits=1, probes=1
        )
        self.assertTrue(one_probe["rejected"])

    def test_nonlinear_bit_frontier_still_needs_fifteen_probes(self) -> None:
        threshold = minimum_adaptive_nonlinear_probes(
            rows=31,
            columns=42,
            cells=proportional_cells(31 * 42),
            alphabet_bits=1,
        )
        self.assertEqual(threshold["minimum_probes_not_degree_rejected"], 15)
        self.assertEqual(
            threshold["last_rejected_witness"]["maximum_matrix_rank"], 16
        )

    def test_nonlinear_words_reduce_capacity_threshold_but_not_construct(self) -> None:
        case = proportional_nonlinear_word_case(rows=31, columns=42)
        self.assertEqual(case["minimum_probes_not_degree_rejected"], 2)
        self.assertEqual(
            Fraction(case["favorable_traffic_fraction"]), Fraction(16, 651)
        )
        self.assertTrue(case["registered_target_rejected"])
        self.assertFalse(case["constructor_exists"])

    def test_scan_identifies_first_open_capacity_point(self) -> None:
        scan = scan_proportional_nonlinear_words(maximum_side=128)
        self.assertEqual(scan["rectangles_checked"], 8_256)
        self.assertEqual(scan["registered_target_passing_count"], 4_257)
        first = scan["smallest_area_capacity_survivor"]
        self.assertEqual((first["rows"], first["columns"]), (25, 108))
        self.assertEqual(
            Fraction(first["favorable_traffic_fraction"]), Fraction(8, 675)
        )
        best = scan["best_traffic_capacity_case"]
        self.assertEqual((best["rows"], best["columns"]), (128, 128))
        self.assertEqual(
            Fraction(best["favorable_traffic_fraction"]), Fraction(1, 256)
        )

    def test_audit_keeps_capacity_survivors_unconstructed(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertTrue(boundary["arbitrary_nonlinear_block_local_word_cells_covered"])
        self.assertFalse(boundary["capacity_survivor_is_a_construction"])
        self.assertFalse(boundary["global_cross_matrix_encoding_covered"])
        self.assertFalse(boundary["surviving_runtime_candidate"])


if __name__ == "__main__":
    unittest.main()
