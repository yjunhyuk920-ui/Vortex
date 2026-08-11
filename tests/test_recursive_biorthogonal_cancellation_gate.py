from __future__ import annotations

from fractions import Fraction
import unittest

from vortex_runtime.recursive_biorthogonal_cancellation_gate import (
    DECISION,
    derive_audit,
    first_unclosed_by_recursive_cancellation_gates,
    recursive_biorthogonal_cancellation_case,
)


class RecursiveBiorthogonalCancellationGateTests(unittest.TestCase):
    def test_pair_peeling_accumulates_cancellation(self) -> None:
        result = recursive_biorthogonal_cancellation_case(
            rows=31, columns=43, atoms=1559, rank_one_radius=15
        )
        by_rank = {case["rank"]: case for case in result["checked_cases"]}
        self.assertEqual(by_rank[18]["forced_cancellation"], 2)
        self.assertEqual(by_rank[20]["forced_cancellation"], 4)
        self.assertEqual(by_rank[22]["forced_cancellation"], 6)
        self.assertEqual(
            by_rank[22]["proof_source"],
            "RECURSIVE_ADJACENT_PAIR_PEELING",
        )

    def test_prior_frontier_is_rejected_at_rank_twenty_two(self) -> None:
        result = recursive_biorthogonal_cancellation_case(
            rows=31, columns=43, atoms=1559, rank_one_radius=15
        )
        violation = result["first_violation"]
        self.assertTrue(result["recursive_cancellation_rejects"])
        self.assertEqual(violation["rank"], 22)
        self.assertEqual(violation["forced_cancellation"], 6)
        self.assertEqual(violation["forced_radius_at_most_rank"], 324)
        self.assertLess(Fraction(violation["ball_to_determinantal_ratio"]), 1)

    def test_thirty_two_by_forty_two_is_also_rejected(self) -> None:
        result = recursive_biorthogonal_cancellation_case(
            rows=32, columns=42, atoms=1572, rank_one_radius=15
        )
        self.assertTrue(result["recursive_cancellation_rejects"])

    def test_first_combined_survivor_moves_to_thirty_one_by_forty_two(self) -> None:
        result = first_unclosed_by_recursive_cancellation_gates()
        self.assertEqual(result["exact_subset_slack_rank"], 857)
        self.assertEqual(result["first_unclosed_shape"], [31, 42])
        self.assertEqual(result["first_unclosed_atoms"], 1523)
        self.assertEqual(result["first_unclosed_radius"], 15)
        self.assertEqual(
            result["preceding_rejection_counts"][
                "RECURSIVE_BIORTHOGONAL_CANCELLATION"
            ],
            7,
        )

    def test_audit_keeps_nonlinear_decoder_models_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertTrue(boundary["first_856_capacity_ordered_shapes_rejected"])
        self.assertFalse(boundary["thirty_one_by_forty_two_constructed"])
        self.assertFalse(boundary["adaptive_addresses_covered"])
        self.assertIn("NO_SURVIVING_CANDIDATE", result["decisions"])


if __name__ == "__main__":
    unittest.main()

