from __future__ import annotations

from fractions import Fraction
import unittest

from vortex_runtime.segre_sparse_cover_fourier_gate import (
    DECISION,
    derive_audit,
    distinct_rectangular_rank_one_masks,
    fourier_cover_case,
    first_unclosed_by_subset_slack,
    hamming_ball_character_sum,
    kernel_defect_requirement,
    krawtchouk,
    maximum_equal_ordered_pairs,
    rank_one_fourier_coefficient,
)


class SegreSparseCoverFourierGateTests(unittest.TestCase):
    def test_rank_one_count_and_transform_match_exhaustive_two_by_two(self) -> None:
        self.assertEqual(distinct_rectangular_rank_one_masks(2, 2), 10)
        self.assertEqual(
            rank_one_fourier_coefficient(
                rows=2, columns=2, dual_rank=0, include_zero=True
            ),
            10,
        )
        self.assertEqual(
            rank_one_fourier_coefficient(
                rows=2, columns=2, dual_rank=1, include_zero=True
            ),
            2,
        )
        self.assertEqual(
            rank_one_fourier_coefficient(
                rows=2, columns=2, dual_rank=2, include_zero=True
            ),
            -2,
        )

    def test_krawtchouk_ball_sum_matches_direct_character_count(self) -> None:
        length = 7
        radius = 3
        dual_weight = 2
        direct = 0
        dual_mask = (1 << dual_weight) - 1
        for subset in range(1 << length):
            if subset.bit_count() <= radius:
                direct += (-1) ** ((subset & dual_mask).bit_count() & 1)
        self.assertEqual(krawtchouk(2, dual_weight, length), 1)
        self.assertEqual(
            hamming_ball_character_sum(
                length=length, radius=radius, dual_weight=dual_weight
            ),
            direct,
        )

    def test_spanning_duplicate_bound_charges_all_extra_copies(self) -> None:
        self.assertEqual(
            maximum_equal_ordered_pairs(atoms=619, ambient_dimension=529),
            8_809,
        )

    def test_registered_square_has_exact_extreme_weight_bands(self) -> None:
        result = fourier_cover_case(rows=23, columns=23)
        self.assertEqual(result["atoms"], 619)
        self.assertEqual(result["radius"], 6)
        self.assertEqual(
            result["valid_rank_one_dual_weight_ranges"],
            [[30, 46], [574, 590]],
        )
        self.assertEqual(result["minimum_absolute_character_bias"], 527)
        self.assertEqual(result["maximum_equal_ordered_atom_pairs"], 8_809)
        self.assertEqual(
            result["maximum_nonzero_difference_rank_one_correlation"],
            "35184355311617/70368727400449",
        )
        self.assertTrue(result["fourier_second_moment_rejects"])
        self.assertGreater(Fraction(result["second_moment_gap"]), 0)

    def test_other_near_capacity_rectangles_are_rejected(self) -> None:
        for rows, columns in (
            (14, 56),
            (21, 25),
            (22, 24),
            (15, 47),
            (12, 122),
            (13, 81),
            (17, 37),
            (14, 64),
        ):
            with self.subTest(shape=(rows, columns)):
                self.assertTrue(
                    fourier_cover_case(
                        rows=rows, columns=columns
                    )["fourier_second_moment_rejects"]
                )

    def test_exact_subset_slack_scan_reaches_thirteen_by_eighty_nine(self) -> None:
        result = first_unclosed_by_subset_slack()
        self.assertEqual(result["exact_subset_slack_rank"], 10)
        self.assertEqual(len(result["preceding_rejected_shapes"]), 9)
        first = result["first_unclosed_case"]
        self.assertEqual((first["rows"], first["columns"]), (13, 89))
        self.assertFalse(first["fourier_second_moment_rejects"])

    def test_first_declared_slack_case_is_not_overclaimed(self) -> None:
        result = fourier_cover_case(rows=13, columns=89)
        self.assertEqual(result["atoms"], 1_353)
        self.assertEqual(result["radius"], 13)
        self.assertEqual(result["minimum_absolute_character_bias"], 1)
        self.assertFalse(result["fourier_second_moment_rejects"])
        self.assertEqual(result["rejection_reason"], "SECOND_MOMENT_NOT_DECISIVE")

    def test_any_slack_survivor_needs_short_kernel_defects(self) -> None:
        result = kernel_defect_requirement(
            rows=13, columns=89, atoms=1_353, radius=13
        )
        self.assertTrue(result["condition_applies"])
        self.assertEqual(result["required_kernel_minimum_distance_at_most"], 39)
        self.assertFalse(result["random_high_girth_dictionary_can_be_promoted"])

    def test_audit_keeps_uncovered_models_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertFalse(boundary["all_rectangular_shapes_rejected"])
        self.assertFalse(boundary["thirteen_by_eighty_nine_constructed"])
        self.assertFalse(boundary["adaptive_addresses_covered"])
        self.assertIn("NO_SURVIVING_CANDIDATE", result["decisions"])


if __name__ == "__main__":
    unittest.main()
