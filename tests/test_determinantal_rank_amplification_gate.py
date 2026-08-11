from __future__ import annotations

from fractions import Fraction
import unittest

from vortex_runtime.determinantal_rank_amplification_gate import (
    DECISION,
    binary_determinantal_count,
    binary_matrix_rank_count,
    derive_audit,
    first_unclosed_by_amplification_and_averaging,
    maximum_rank_one_points,
    mixed_weight_representative_case,
    rank_amplification_case,
)
from vortex_runtime.joint_batch_coset_geometry import (
    maximum_rank_one_points_control,
)


class DeterminantalRankAmplificationGateTests(unittest.TestCase):
    def test_binary_two_by_two_rank_counts_are_exact(self) -> None:
        self.assertEqual(binary_matrix_rank_count(rows=2, columns=2, rank=0), 1)
        self.assertEqual(binary_matrix_rank_count(rows=2, columns=2, rank=1), 9)
        self.assertEqual(binary_matrix_rank_count(rows=2, columns=2, rank=2), 6)
        self.assertEqual(
            binary_determinantal_count(rows=2, columns=2, maximum_rank=2), 16
        )

    def test_closed_rank_one_intersection_matches_all_small_exact_controls(self) -> None:
        for rows in range(1, 5):
            for columns in range(rows, 7):
                for dimension in range(rows * columns + 1):
                    with self.subTest(shape=(rows, columns), dimension=dimension):
                        self.assertEqual(
                            maximum_rank_one_points(
                                rows=rows,
                                columns=columns,
                                subspace_dimension=dimension,
                            ),
                            maximum_rank_one_points_control(
                                rows, columns, dimension
                            ),
                        )

    def test_seventeen_by_forty_three_fails_rank_two_capacity(self) -> None:
        result = rank_amplification_case(
            rows=17, columns=43, atoms=855, rank_one_radius=8
        )
        violation = result["first_violation"]
        self.assertTrue(result["rank_amplification_rejects"])
        self.assertEqual(violation["maximum_rank"], 2)
        self.assertEqual(violation["amplified_radius"], 16)
        self.assertEqual(
            violation["hamming_ball_size"],
            "3450282117207738592915369522984017",
        )
        self.assertEqual(
            violation["determinantal_count"],
            "221532928720799928211216887624892418",
        )

    def test_balanced_rank_amplification_survivor_fails_mixed_weights(self) -> None:
        result = mixed_weight_representative_case(
            rows=32, columns=39, atoms=1460, radius=14
        )
        self.assertEqual(result["witness_sampled_atoms"], 775)
        self.assertTrue(result["mixed_weight_averaging_rejects"])
        self.assertLess(Fraction(result["upper_to_required_ratio"]), 1)

    def test_first_combined_survivor_is_thirty_by_forty(self) -> None:
        result = first_unclosed_by_amplification_and_averaging()
        self.assertEqual(result["exact_subset_slack_rank"], 771)
        self.assertEqual(result["first_unclosed_shape"], [30, 40])
        self.assertEqual(result["first_unclosed_atoms"], 1404)
        self.assertEqual(result["first_unclosed_radius"], 14)
        self.assertEqual(
            result["preceding_rejection_counts"]["MIXED_WEIGHT_AVERAGING"],
            4,
        )

    def test_declared_survivor_passes_both_new_necessities_only(self) -> None:
        amplified = rank_amplification_case(
            rows=30, columns=40, atoms=1404, rank_one_radius=14
        )
        averaged = mixed_weight_representative_case(
            rows=30, columns=40, atoms=1404, radius=14
        )
        self.assertFalse(amplified["rank_amplification_rejects"])
        self.assertEqual(amplified["closest_checked_case"]["maximum_rank"], 22)
        self.assertGreater(
            Fraction(amplified["closest_checked_case"]["ball_to_determinantal_ratio"]),
            1,
        )
        self.assertEqual(averaged["witness_sampled_atoms"], 795)
        self.assertFalse(averaged["mixed_weight_averaging_rejects"])
        self.assertGreater(Fraction(averaged["upper_to_required_ratio"]), 1)

    def test_audit_keeps_uncovered_models_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertTrue(boundary["first_770_capacity_ordered_shapes_rejected"])
        self.assertFalse(boundary["thirty_by_forty_constructed"])
        self.assertFalse(boundary["adaptive_addresses_covered"])
        self.assertIn("NO_SURVIVING_CANDIDATE", result["decisions"])


if __name__ == "__main__":
    unittest.main()
