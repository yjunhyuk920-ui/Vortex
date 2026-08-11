from __future__ import annotations

from fractions import Fraction
import unittest

from vortex_runtime.fourier_fiber_direct_sum_frontier import (
    BINARY_RADIUS_WITNESS,
    DECISION,
    REGISTERED_PARAMETERS,
    all_linear_entropy_witness,
    binary_entropy_decimal,
    character_dimension_method_ceiling,
    derive_audit,
    exhaustive_binary_fiber_control,
    hamming_ball_size,
    registered_rank_one_query_log2_upper_bound,
    walsh_row_orthogonality_control,
)


class FourierFiberDirectSumFrontierTests(unittest.TestCase):
    def test_small_hamming_ball_values(self) -> None:
        self.assertEqual(hamming_ball_size(3, 0), 1)
        self.assertEqual(hamming_ball_size(3, 1), 4)
        self.assertEqual(hamming_ball_size(3, 2), 7)
        self.assertEqual(hamming_ball_size(3, 3), 8)

    def test_entropy_witness_is_strict_and_finite(self) -> None:
        result = all_linear_entropy_witness(
            data_bits=REGISTERED_PARAMETERS,
            advice_bits=8 * (8 * (1 << 30)),
            radius=BINARY_RADIUS_WITNESS,
        )
        self.assertEqual(result["radius_witness"], "131/500")
        self.assertGreater(float(result["strict_entropy_margin"]), 0)
        expected = (131 * REGISTERED_PARAMETERS) // 500 + 1
        self.assertEqual(
            result["minimum_worst_case_raw_bit_probes"], expected
        )
        self.assertGreater(result["minimum_probe_percent"], 26.2 - 1e-9)

    def test_exhaustive_binary_fibers_satisfy_volume_bound(self) -> None:
        result = exhaustive_binary_fiber_control(dimension=3)
        self.assertEqual(result["nonempty_fibers"], 255)
        self.assertEqual(result["parity_query_checks"], 2_040)
        self.assertEqual(result["fiber_volume_violations"], 0)
        self.assertTrue(result["all_fibers_satisfy_bound"])

    def test_walsh_rows_are_orthogonal(self) -> None:
        result = walsh_row_orthogonality_control(dimension=4)
        self.assertEqual(result["distinct_row_pairs"], 120)
        self.assertEqual(result["orthogonality_failures"], 0)
        self.assertTrue(result["restricted_rows_are_independent"])

    def test_registered_rank_one_query_count(self) -> None:
        result = registered_rank_one_query_log2_upper_bound()
        self.assertEqual(
            result["parameters_reconstructed_from_shapes"],
            REGISTERED_PARAMETERS,
        )
        self.assertEqual(
            result["one_independent_tuple_log2_query_count_upper_bound"],
            39_254_528,
        )
        self.assertEqual(
            result["thirty_two_tuple_log2_query_count_upper_bound"],
            1_256_144_896,
        )

    def test_character_dimension_method_ceiling_is_not_algorithm(self) -> None:
        result = character_dimension_method_ceiling(
            data_bits=REGISTERED_PARAMETERS,
            query_log2_count_upper_bound=1_256_144_896,
        )
        self.assertEqual(result["method_ceiling_probe_count"], 114_194_991)
        self.assertEqual(result["power_of_two_ratio_witness"], 11)
        self.assertGreaterEqual(
            result["certified_log2_binomial_lower_bound"],
            result["query_log2_count_upper_bound"],
        )
        self.assertFalse(result["is_algorithmic_query_upper_bound"])
        self.assertFalse(result["is_target_lower_bound"])

    def test_audit_keeps_rank_one_gap_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        all_linear = result["registered_all_linear_tuple"]
        self.assertFalse(
            all_linear[
                "reaches_requested_dfloat_block_under_one_bit_per_probe"
            ]
        )
        self.assertGreater(
            all_linear["forced_over_registered_p50_coefficient_budget"],
            20,
        )
        boundary = result["claim_boundary"]
        self.assertTrue(boundary["charges_global_advice_synergy"])
        self.assertFalse(boundary["rank_one_query_family_theorem"])
        self.assertFalse(boundary["target_scale_rank_one_lower_bound"])

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            binary_entropy_decimal(Fraction(0, 1))
        with self.assertRaises(ValueError):
            all_linear_entropy_witness(
                data_bits=4, advice_bits=4, radius=Fraction(1, 4)
            )
        with self.assertRaises(ValueError):
            hamming_ball_size(3, 4)
        with self.assertRaises(ValueError):
            character_dimension_method_ceiling(
                data_bits=0, query_log2_count_upper_bound=1
            )
        with self.assertRaises(ValueError):
            exhaustive_binary_fiber_control(dimension=5)


if __name__ == "__main__":
    unittest.main()
