from __future__ import annotations

import unittest

from vortex_runtime.joint_batch_coset_geometry import (
    maximum_rank_one_points_control,
)
from vortex_runtime.nonlinear_router_cover_gate import (
    DECISION,
    adaptive_nonlinear_depth_cover_case,
    all_query_union_descriptor_count,
    affine_first_adaptive_cover_case,
    derive_audit,
    first_unclosed_by_area,
    independent_query_batch_union_floor,
    maximum_rank_one_points_in_subspace,
    maximum_rank_one_points_closed_form,
    minimum_worst_source_all_query_union_cells,
    nonadaptive_nonlinear_depth_cover_case,
    nonadaptive_nonlinear_pair_cover_case,
    one_value_stage_adaptive_cover_case,
    scan_side_bound,
    rank_one_mask_count,
)


class NonlinearRouterCoverGateTests(unittest.TestCase):
    def test_rank_one_count(self) -> None:
        self.assertEqual(rank_one_mask_count(2, 3), 21)

    def test_deficit_dp_matches_existing_exact_product_code_controls(self) -> None:
        for rows, columns in ((2, 2), (2, 3), (2, 4), (3, 3), (3, 4)):
            for dimension in range(rows * columns + 1):
                expected = maximum_rank_one_points_control(
                    rows, columns, dimension
                )
                actual = maximum_rank_one_points_in_subspace(
                    rows, columns, dimension
                )["maximum_rank_one_points"]
                self.assertEqual(actual, expected)
                self.assertEqual(
                    maximum_rank_one_points_closed_form(
                        rows, columns, dimension
                    ),
                    expected,
                )

    def test_transpose_invariance(self) -> None:
        for dimension in range(9):
            left = maximum_rank_one_points_in_subspace(3, 5, dimension)
            right = maximum_rank_one_points_in_subspace(5, 3, dimension)
            self.assertEqual(
                left["maximum_rank_one_points"], right["maximum_rank_one_points"]
            )

    def test_registered_intersection_is_exact(self) -> None:
        result = maximum_rank_one_points_in_subspace(25, 108, 128)
        self.assertEqual(
            result["maximum_rank_one_points"], (1 << 108) + (1 << 21) - 3
        )
        self.assertEqual(result["maximizing_deficits"][-2:], [20, 108])
        self.assertTrue(all(value == 0 for value in result["maximizing_deficits"][:-2]))

    def test_nonadaptive_arbitrary_nonlinear_words_rejected(self) -> None:
        result = nonadaptive_nonlinear_pair_cover_case(
            rows=25, columns=108, cells=50, word_bits=64
        )
        self.assertEqual(result["address_pairs_including_singletons"], 1275)
        self.assertTrue(result["cell_values_may_be_arbitrary_nonlinear"])
        self.assertTrue(result["rejected"])
        self.assertLess(result["coverage_ratio"], 0.00004)

    def test_affine_first_adaptive_arbitrary_second_rejected(self) -> None:
        result = affine_first_adaptive_cover_case(
            rows=25, columns=108, cells=50, word_bits=64
        )
        self.assertEqual(result["route_groups_at_most"], 2500)
        self.assertTrue(result["second_word_may_be_arbitrary_nonlinear"])
        self.assertTrue(result["second_address_may_depend_on_first_value"])
        self.assertTrue(result["rejected"])
        self.assertLess(result["coverage_ratio"], 0.000075)

    def test_fully_nonlinear_adaptive_two_probe_rejected(self) -> None:
        result = adaptive_nonlinear_depth_cover_case(
            rows=25, columns=108, cells=50, word_bits=64, probes=2
        )
        self.assertTrue(result["stored_cells_may_be_arbitrary_nonlinear"])
        self.assertTrue(result["every_address_after_the_first_may_be_value_adaptive"])
        self.assertTrue(result["decoder_may_be_arbitrary_deterministic_logic"])
        self.assertTrue(result["rejected"])
        self.assertLess(result["coverage_ratio_decimal"], 0.000075)

    def test_side_128_scan_has_no_target_word_survivor(self) -> None:
        result = scan_side_bound(128)
        self.assertEqual(result["rectangles_checked"], 8256)
        self.assertEqual(result["unclosed_count"], 0)
        self.assertEqual(
            result["zero_64bit_probe_budget"] + result["adaptive_cover_rejected"],
            8256,
        )

    def test_first_unclosed_area_is_5400(self) -> None:
        result = first_unclosed_by_area(5400)
        self.assertEqual(result["first_unclosed_area"], 5400)
        shapes = sorted(
            (case["rows"], case["columns"])
            for case in result["first_unclosed_cases"]
        )
        self.assertEqual(shapes, [(24, 225), (25, 216)])
        for case in result["first_unclosed_cases"]:
            self.assertEqual(case["padded_words"], 99)
            self.assertEqual(case["maximum_target_probes"], 4)

    def test_first_frontier_nonadaptive_four_probe_is_rejected(self) -> None:
        for rows, columns in ((24, 225), (25, 216)):
            result = nonadaptive_nonlinear_depth_cover_case(
                rows=rows, columns=columns, cells=99, word_bits=64, probes=4
            )
            self.assertTrue(result["stored_cells_may_be_arbitrary_nonlinear"])
            self.assertFalse(result["value_adaptive_addresses"])
            self.assertTrue(result["rejected"])
            self.assertLess(result["coverage_ratio_decimal"], 1.0)

    def test_first_frontier_one_value_stage_is_rejected(self) -> None:
        for rows, columns in ((24, 225), (25, 216)):
            result = one_value_stage_adaptive_cover_case(
                rows=rows, columns=columns, cells=99, word_bits=64, probes=4
            )
            self.assertTrue(result["first_value_may_select_the_remaining_payload_set"])
            self.assertFalse(result["later_payload_values_may_change_addresses"])
            self.assertTrue(result["rejected"])
            self.assertLess(result["coverage_ratio_decimal"], 1.0)

    def test_all_query_union_descriptor_thresholds(self) -> None:
        old = minimum_worst_source_all_query_union_cells(
            source_bits=2700, cells=50, word_bits=64
        )
        self.assertEqual(old["forced_worst_source_all_query_union_cells"], 42)
        self.assertLess(
            all_query_union_descriptor_count(
                cells=50, word_bits=64, maximum_union_cells=41
            ),
            1 << 2700,
        )

        frontier = minimum_worst_source_all_query_union_cells(
            source_bits=5400, cells=99, word_bits=64
        )
        self.assertEqual(frontier["forced_worst_source_all_query_union_cells"], 84)
        self.assertLess(
            all_query_union_descriptor_count(
                cells=99, word_bits=64, maximum_union_cells=83
            ),
            1 << 5400,
        )
        self.assertGreaterEqual(
            all_query_union_descriptor_count(
                cells=99, word_bits=64, maximum_union_cells=84
            ),
            1 << 5400,
        )

    def test_area5400_strong_32_query_union_is_eight_times_target(self) -> None:
        for rows, columns in ((24, 225), (25, 216)):
            result = independent_query_batch_union_floor(
                rows=rows,
                columns=columns,
                cells=99,
                word_bits=64,
                batch_queries=32,
            )
            self.assertEqual(
                result["all_query_union_capacity"][
                    "forced_worst_source_all_query_union_cells"
                ],
                84,
            )
            self.assertEqual(result["forced_distinct_cells_in_some_query_tuple"], 32)
            self.assertEqual(result["forced_distinct_encoded_bits"], 2048)
            self.assertEqual(result["forced_union_fraction"], "64/675")
            self.assertEqual(result["target_multiple"], "8")
            self.assertTrue(result["target_rejected_under_strong_query_interface"])
            self.assertFalse(
                result["ordinary_transformer_causal_reachability_of_hard_tuple"]
            )

    def test_claim_boundary_records_stronger_post_preregistration_gate(self) -> None:
        payload = derive_audit()
        self.assertEqual(payload["decision"], DECISION)
        self.assertTrue(
            payload["claim_boundary"]["fully_nonlinear_adaptive_first_word_covered"]
        )
        self.assertTrue(
            payload["claim_boundary"]["side_one_to_128_target_word_router_closed"]
        )
        self.assertTrue(
            payload["claim_boundary"][
                "first_unclosed_requires_multiple_successive_value_adaptive_routing_stages"
            ]
        )
        self.assertTrue(
            payload["claim_boundary"][
                "area5400_strong_independent_32_query_union_target_rejected"
            ]
        )
        self.assertFalse(
            payload["claim_boundary"][
                "ordinary_transformer_causal_reachability_of_hard_32_tuple"
            ]
        )
        self.assertFalse(payload["claim_boundary"]["target_candidate_constructed"])
        self.assertEqual(payload["obligation_update"]["O1"], "OPEN")
        self.assertEqual(payload["obligation_update"]["O5"], "OPEN")

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            maximum_rank_one_points_in_subspace(0, 2, 1)
        with self.assertRaises(ValueError):
            maximum_rank_one_points_in_subspace(2, 2, 5)
        with self.assertRaises(ValueError):
            nonadaptive_nonlinear_pair_cover_case(
                rows=2, columns=2, cells=0, word_bits=64
            )
        with self.assertRaises(ValueError):
            affine_first_adaptive_cover_case(
                rows=2, columns=2, cells=4, word_bits=0
            )


if __name__ == "__main__":
    unittest.main()
