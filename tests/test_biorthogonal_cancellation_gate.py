from __future__ import annotations

from fractions import Fraction
import unittest

import numpy as np

from vortex_runtime.biorthogonal_cancellation_gate import (
    DECISION,
    anti_flag_graph_parameters,
    biorthogonal_cancellation_case,
    derive_audit,
    first_unclosed_by_all_sparse_cover_gates,
    minimum_cancellation_from_pair_intersections,
    minimum_distinct_nonzero_support_weight,
)


def _parity(value: int) -> int:
    return value.bit_count() & 1


def _anti_flag_adjacency(rank: int) -> np.ndarray:
    vertices = [
        (left, right)
        for left in range(1, 1 << rank)
        for right in range(1 << rank)
        if _parity(left & right) == 1
    ]
    adjacency = np.zeros((len(vertices), len(vertices)), dtype=np.float64)
    for first, (left, right) in enumerate(vertices):
        for second in range(first + 1, len(vertices)):
            other_left, other_right = vertices[second]
            if (
                _parity(right & other_left) == 0
                and _parity(other_right & left) == 0
            ):
                adjacency[first, second] = 1
                adjacency[second, first] = 1
    return adjacency


class BiorthogonalCancellationGateTests(unittest.TestCase):
    def test_rank_four_anti_flag_spectrum_control(self) -> None:
        parameters = anti_flag_graph_parameters(4)
        adjacency = _anti_flag_adjacency(4)
        eigenvalues = np.linalg.eigvalsh(adjacency)
        self.assertEqual(adjacency.shape, (120, 120))
        self.assertTrue(np.all(adjacency.sum(axis=1) == parameters["degree"]))
        self.assertAlmostEqual(eigenvalues[0], -8.0, places=8)
        self.assertEqual(parameters["least_eigenvalue_magnitude"], 8)

    def test_distinct_support_activity_fills_light_shells_first(self) -> None:
        result = minimum_distinct_nonzero_support_weight(
            length=6, count=8, radius=2
        )
        self.assertEqual(result["minimum_total_weight"], 10)
        self.assertEqual(result["minimum_average_weight"], Fraction(5, 4))
        self.assertEqual(result["shell_counts"][0]["selected"], 6)
        self.assertEqual(result["shell_counts"][1]["selected"], 2)

    def test_pair_intersection_inversion_is_exact_at_first_boundaries(self) -> None:
        self.assertEqual(
            minimum_cancellation_from_pair_intersections(
                rank=22, pair_intersections=3, maximum_cancellation=308
            ),
            2,
        )
        self.assertEqual(
            minimum_cancellation_from_pair_intersections(
                rank=22, pair_intersections=4, maximum_cancellation=308
            ),
            4,
        )
        self.assertEqual(
            minimum_cancellation_from_pair_intersections(
                rank=22, pair_intersections=10, maximum_cancellation=308
            ),
            4,
        )
        self.assertEqual(
            minimum_cancellation_from_pair_intersections(
                rank=22, pair_intersections=11, maximum_cancellation=308
            ),
            6,
        )

    def test_thirty_by_forty_is_rejected_at_rank_twenty_two(self) -> None:
        result = biorthogonal_cancellation_case(
            rows=30, columns=40, atoms=1404, rank_one_radius=14
        )
        violation = result["first_violation"]
        self.assertTrue(result["biorthogonal_cancellation_rejects"])
        self.assertEqual(violation["rank"], 22)
        self.assertEqual(violation["guaranteed_pair_intersections"], 4)
        self.assertEqual(violation["forced_cancellation"], 4)
        self.assertEqual(violation["forced_radius_at_most_rank"], 304)
        self.assertLess(Fraction(violation["ball_to_determinantal_ratio"]), 1)

    def test_next_balanced_shape_is_also_rejected(self) -> None:
        result = biorthogonal_cancellation_case(
            rows=31, columns=39, atoms=1414, rank_one_radius=14
        )
        self.assertTrue(result["biorthogonal_cancellation_rejects"])
        violation = result["first_violation"]
        self.assertEqual(violation["rank"], 22)
        self.assertAlmostEqual(
            violation["ball_to_determinantal_ratio_decimal"],
            0.5723204167945867,
        )

    def test_first_combined_survivor_moves_to_thirty_one_by_forty_three(self) -> None:
        result = first_unclosed_by_all_sparse_cover_gates()
        self.assertEqual(result["exact_subset_slack_rank"], 847)
        self.assertEqual(result["first_unclosed_shape"], [31, 43])
        self.assertEqual(result["first_unclosed_atoms"], 1559)
        self.assertEqual(result["first_unclosed_radius"], 15)
        self.assertEqual(
            result["preceding_rejection_counts"]["BIORTHOGONAL_CANCELLATION"],
            3,
        )

    def test_audit_keeps_uncovered_decoder_models_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertTrue(boundary["first_846_capacity_ordered_shapes_rejected"])
        self.assertFalse(boundary["thirty_one_by_forty_three_constructed"])
        self.assertFalse(boundary["adaptive_addresses_covered"])
        self.assertIn("NO_SURVIVING_CANDIDATE", result["decisions"])


if __name__ == "__main__":
    unittest.main()
