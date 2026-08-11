from __future__ import annotations

from fractions import Fraction
from functools import reduce
import itertools
from operator import xor
import unittest

from vortex_runtime.segre_ruling_preimage_sphere_gate import (
    DECISION,
    derive_audit,
    first_unclosed_by_combined_gates,
    low_weight_subspace_upper,
    projected_ruling_cover_requirement,
    restricted_tensor_preimage_case,
    strongest_restricted_tensor_case,
)


class SegreRulingPreimageSphereGateTests(unittest.TestCase):
    def test_information_set_bound_matches_small_systematic_enumeration(self) -> None:
        # Generator [I_3 | A].  Every ambient low-weight word maps to a
        # distinct projected low-weight word, as used by the proof.
        rows = (0b100101, 0b010110, 0b001111)
        weights = [
            reduce(xor, (row for bit, row in zip(bits, rows) if bit), 0).bit_count()
            for bits in itertools.product((0, 1), repeat=3)
        ]
        self.assertLessEqual(
            sum(weight <= 2 for weight in weights),
            low_weight_subspace_upper(dimension=3, radius=2),
        )

    def test_prior_thirteen_by_eighty_nine_survivor_is_rejected_exactly(self) -> None:
        result = restricted_tensor_preimage_case(
            rows=13,
            columns=89,
            atoms=1_353,
            radius=13,
            left_dimension=1,
            right_dimension=89,
        )
        self.assertEqual(result["kernel_dimension"], 196)
        self.assertEqual(result["preimage_dimension"], 285)
        self.assertEqual(result["low_weight_subspace_upper"], "10451278437645598672024")
        self.assertEqual(result["required_low_weight_words"], "618970019642690137449562112")
        self.assertTrue(result["preimage_sphere_rejects"])
        self.assertLess(Fraction(result["upper_to_required_ratio"]), 1)

    def test_strongest_prior_witness_is_the_large_ruling(self) -> None:
        result = strongest_restricted_tensor_case(rows=13, columns=89)
        self.assertEqual((result["left_dimension"], result["right_dimension"]), (1, 89))
        self.assertTrue(result["preimage_sphere_rejects"])

    def test_first_combined_survivor_moves_to_eighteen_by_thirty_six(self) -> None:
        result = first_unclosed_by_combined_gates()
        self.assertEqual(result["exact_subset_slack_rank"], 11)
        self.assertEqual(len(result["preceding_rejected_shapes"]), 10)
        self.assertEqual(result["first_unclosed_shape"], [18, 36])
        thirteen = result["preceding_rejected_shapes"][-1]
        self.assertEqual((thirteen["rows"], thirteen["columns"]), (13, 89))
        self.assertEqual(thirteen["rejected_by"], "PREIMAGE_SPHERE")

    def test_survivor_implies_exact_projected_covering_code_parameters(self) -> None:
        result = projected_ruling_cover_requirement(
            rows=18, columns=36, atoms=758, radius=7
        )
        self.assertEqual(result["projected_code_length"], 146)
        self.assertEqual(result["projected_code_dimension"], 110)
        self.assertEqual(result["projected_code_codimension"], 36)
        self.assertEqual(result["sphere_volume"], "255108299740")
        self.assertEqual(result["syndrome_count"], "68719476736")
        self.assertTrue(result["ordinary_covering_code_requirement_passes_sphere_bound"])
        self.assertFalse(result["ordinary_covering_code_constructed"])

    def test_survivor_has_no_restricted_tensor_sphere_violation(self) -> None:
        result = strongest_restricted_tensor_case(rows=18, columns=36)
        self.assertEqual((result["left_dimension"], result["right_dimension"]), (18, 36))
        self.assertEqual(result["preimage_dimension"], 758)
        self.assertFalse(result["preimage_sphere_rejects"])

    def test_audit_keeps_uncovered_models_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertTrue(boundary["thirteen_by_eighty_nine_rejected"])
        self.assertFalse(boundary["eighteen_by_thirty_six_constructed"])
        self.assertFalse(boundary["adaptive_addresses_covered"])
        self.assertIn("NO_SURVIVING_CANDIDATE", result["decisions"])


if __name__ == "__main__":
    unittest.main()
