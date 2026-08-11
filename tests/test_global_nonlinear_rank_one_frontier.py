from __future__ import annotations

from fractions import Fraction
import unittest

from vortex_runtime.global_nonlinear_rank_one_frontier import (
    DECISION,
    exact_subrectangle_summary_lower_bound,
    derive_audit,
    full_sweep_batching_metric_boundary,
    larsen_williams_leading_terms,
    rank_one_three_query_dependency,
    scalarization_of_static_matvec_lower_bound,
)


class GlobalNonlinearRankOneFrontierTests(unittest.TestCase):
    def test_larsen_williams_leading_terms_are_six_point_two_five_percent(
        self,
    ) -> None:
        result = larsen_williams_leading_terms()
        self.assertEqual(result["matrix_bytes"], 33_554_432)
        self.assertEqual(result["leading_probe_words"], 262_144)
        self.assertEqual(result["leading_probe_bytes"], 2_097_152)
        self.assertEqual(result["leading_redundancy_bits"], 16_777_216)
        self.assertEqual(result["leading_redundancy_bytes"], 2_097_152)
        self.assertEqual(
            Fraction(result["leading_probe_byte_fraction_of_one_bit_matrix"]),
            Fraction(1, 16),
        )
        self.assertEqual(
            Fraction(result["leading_redundancy_fraction_of_one_bit_matrix"]),
            Fraction(1, 16),
        )
        self.assertTrue(result["published_bounds_have_hidden_constants"])
        self.assertFalse(
            result["exact_parity_count_signed_sum_or_native_numeric_output"]
        )
        self.assertFalse(result["target_certificate"])

    def test_rank_one_query_code_has_a_three_query_dependency(self) -> None:
        result = rank_one_three_query_dependency()
        self.assertEqual(
            result["maximum_k_wise_independence_of_full_distinct_query_family"],
            2,
        )
        self.assertEqual(result["support_size_of_answer_triple_at_most"], 4)
        self.assertEqual(
            result["support_size_required_for_three_wise_independence"], 8
        )
        self.assertEqual(result["strict_threshold_t_times_w_plus_one"], 129)
        self.assertEqual(result["minimum_integer_k_required"], 130)
        self.assertFalse(
            result["premise_can_hold_for_full_rank_one_family"]
        )
        self.assertFalse(result["target_lower_bound_obtained"])

    def test_exact_subrectangle_summary_must_retain_raw_information(self) -> None:
        result = exact_subrectangle_summary_lower_bound(
            rows=7, columns=9, alphabet_size=16
        )
        self.assertEqual(result["cells"], 63)
        self.assertEqual(result["minimum_summary_bits"], 252)
        self.assertFalse(result["compression_below_raw_information_possible"])
        self.assertFalse(
            result["covers_arbitrary_adaptive_global_data_structures"]
        )

    def test_even_ideal_full_matvec_bound_is_weak_after_scalarization(self) -> None:
        result = scalarization_of_static_matvec_lower_bound(
            dimension=16_384,
            whole_vector_probe_lower_bound=16_384**2,
        )
        self.assertEqual(result["implied_scalar_probe_lower_bound"], 16_384)
        self.assertFalse(result["reaches_one_fortieth_of_matrix_words"])

    def test_forty_way_full_sweep_does_not_meet_block_fraction(self) -> None:
        result = full_sweep_batching_metric_boundary(batch_outputs=40)
        self.assertEqual(result["registered_per_token_weight_fraction"], "1/1280")
        self.assertEqual(result["full_sweep_per_output_fraction"], "1/40")
        self.assertEqual(
            result["per_output_over_registered_per_token_ratio"], "32"
        )
        self.assertEqual(
            result["outputs_needed_to_match_weight_io_per_token"], 1280
        )
        self.assertFalse(result["meets_registered_block_read_fraction"])

    def test_audit_keeps_the_general_claim_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertEqual(boundary["zero_rectangle_numeric_lift"], "REJECTED")
        self.assertEqual(
            boundary["kpi_limited_independence_application"],
            "PREMISE FALSE",
        )
        self.assertEqual(
            boundary["universal_2_5_percent_guarantee"],
            "NOT ESTABLISHED",
        )
        self.assertEqual(
            boundary["universal_2_5_percent_impossibility"],
            "NOT PROVED",
        )
        self.assertEqual(boundary["core_candidate"], "NONE")
        self.assertEqual(boundary["model_forward_calls"], 0)
        self.assertEqual(boundary["hardware_actions"], 0)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            larsen_williams_leading_terms(dimension=3)
        with self.assertRaises(ValueError):
            rank_one_three_query_dependency(minimum_even_probes=1)
        with self.assertRaises(ValueError):
            exact_subrectangle_summary_lower_bound(
                rows=1, columns=1, alphabet_size=1
            )
        with self.assertRaises(ValueError):
            scalarization_of_static_matvec_lower_bound(
                dimension=0, whole_vector_probe_lower_bound=1
            )


if __name__ == "__main__":
    unittest.main()
