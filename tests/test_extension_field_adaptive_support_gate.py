from __future__ import annotations

from fractions import Fraction
import unittest

from vortex_runtime.extension_field_adaptive_support_gate import (
    ALL_DFLOAT_STORAGE_TO_ONE_PLANE_EXPANSION,
    DECISION,
    FOUR_Q4_LANE_TRAFFIC_FRACTION,
    aggregate_block_local_case,
    derive_audit,
    exhaustive_gf4_intersection_control,
    minimum_sparse_support_radius,
    proportional_block_case,
    rational_dfloat_rejection_witness,
    sparse_support_union_bound,
)


class ExtensionFieldAdaptiveSupportGateTests(unittest.TestCase):
    def test_support_count_and_minimum_radius_are_exact(self) -> None:
        self.assertEqual(
            sparse_support_union_bound(cells=3, probes=1),
            1 + 3 * 2,
        )
        radius = minimum_sparse_support_radius(directions=4, cells=5)
        self.assertEqual(radius, 2)
        self.assertLess(sparse_support_union_bound(cells=5, probes=1), 1 << 4)
        self.assertGreaterEqual(
            sparse_support_union_bound(cells=5, probes=2), 1 << 4
        )

    def test_small_gf4_intersections_meet_the_theorem(self) -> None:
        control = exhaustive_gf4_intersection_control()
        self.assertTrue(control["passes"])
        self.assertEqual(
            control["maximum_binary_intersection_by_generator_count"],
            {0: 1, 1: 2, 2: 4},
        )

    def test_registered_hidden_dimension_misses_both_targets(self) -> None:
        case = proportional_block_case(16_384)
        self.assertEqual(case["field_cells"], 19_172)
        self.assertEqual(case["minimum_probes"], 3_421)
        self.assertEqual(case["logical_target"]["probes"], 194)
        self.assertEqual(case["four_lane_traffic_target"]["probes"], 776)
        self.assertFalse(case["logical_target"]["count_condition_passes"])
        self.assertFalse(
            case["four_lane_traffic_target"]["count_condition_passes"]
        )

    def test_full_dfloat_storage_grant_still_fails_traffic_relaxation(self) -> None:
        case = aggregate_block_local_case(
            name="test",
            storage_expansion=ALL_DFLOAT_STORAGE_TO_ONE_PLANE_EXPANSION,
            target_fraction=FOUR_Q4_LANE_TRAFFIC_FRACTION,
        )
        self.assertTrue(case["target_rejected"])
        self.assertGreater(
            Fraction(case["minimum_to_target_multiplier"]),
            2,
        )
        witness = rational_dfloat_rejection_witness(
            FOUR_Q4_LANE_TRAFFIC_FRACTION
        )
        self.assertTrue(witness["rejects"])
        self.assertEqual(witness["capacity_exponent_strict_upper"], "352/675")

    def test_audit_keeps_only_uncovered_mechanism_classes_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertTrue(boundary["deterministic_adaptive_addresses_rejected"])
        self.assertTrue(boundary["arbitrary_exact_postprocessing_rejected"])
        self.assertFalse(boundary["nonlinear_stored_cells_covered"])
        self.assertFalse(boundary["cross_matrix_mixed_cells_covered"])
        self.assertFalse(boundary["surviving_runtime_candidate"])


if __name__ == "__main__":
    unittest.main()
