from __future__ import annotations

from fractions import Fraction
import unittest

from vortex_runtime.finite_semiring_preprocessing_frontier import (
    DECISION,
    bf16_add,
    derive_audit,
    ideal_model_wide_substitution,
    minimum_block_symbols_for_fraction,
    minimum_block_symbols_for_dfloat_budget,
    native_fp32_accumulation_counterexample,
    native_rounding_semiring_counterexample,
    fp32_add,
    round_to_bfloat16,
    williams_graph_layout,
)


class FiniteSemiringPreprocessingFrontierTests(unittest.TestCase):
    def test_boolean_largest_square_pointer_payload_is_finite(self) -> None:
        result = williams_graph_layout(
            dimension=16_384, alphabet_bits=1, block_symbols=14
        )
        self.assertEqual(result["groups"], 1_171)
        self.assertEqual(result["patterns_per_group"], 16_384)
        self.assertEqual(result["query_neighbor_pointers"], 1_371_241)
        self.assertEqual(result["query_pointer_payload_bits"], 19_197_374)
        self.assertGreater(result["query_pointer_payload_percent_of_raw"], 7.14)
        self.assertGreater(result["sidecar_pointer_payload_gib"], 36)

    def test_model_wide_q4_and_bf16_payloads_miss_block_budget(self) -> None:
        q4 = ideal_model_wide_substitution(alphabet_bits=4, block_symbols=14)
        bf16 = ideal_model_wide_substitution(
            alphabet_bits=16, block_symbols=14
        )
        self.assertFalse(q4["query_payload_fits_requested_block_budget"])
        self.assertFalse(bf16["query_payload_fits_requested_block_budget"])
        self.assertGreater(float(Fraction(q4["query_over_requested_block_budget"])), 1)
        self.assertGreater(float(Fraction(bf16["query_over_requested_block_budget"])), 4)
        self.assertFalse(q4["sidecar_fits_global_advice"])
        self.assertFalse(bf16["sidecar_fits_global_advice"])

    def test_even_boolean_model_wide_sidecar_misses_eight_gib(self) -> None:
        result = ideal_model_wide_substitution(
            alphabet_bits=1, block_symbols=14
        )
        self.assertTrue(result["query_payload_fits_requested_block_budget"])
        self.assertFalse(result["sidecar_fits_global_advice"])
        self.assertGreater(result["sidecar_over_global_8_gib"], 1_000)
        self.assertFalse(
            result["direct_32_query_no_reuse_fits_requested_block_budget"]
        )
        self.assertGreater(
            float(
                Fraction(
                    result[
                        "direct_32_query_no_reuse_over_requested_block_budget"
                    ]
                )
            ),
            8,
        )

    def test_two_point_five_percent_needs_more_symbols_than_theorem_parameter(
        self,
    ) -> None:
        self.assertEqual(minimum_block_symbols_for_fraction(Fraction(1, 40)), 40)
        self.assertGreater(40, 14)

    def test_dfloat_denominator_changes_required_block_by_alphabet(self) -> None:
        self.assertEqual(
            minimum_block_symbols_for_dfloat_budget(alphabet_bits=1), 4
        )
        self.assertEqual(
            minimum_block_symbols_for_dfloat_budget(alphabet_bits=4), 15
        )
        self.assertEqual(
            minimum_block_symbols_for_dfloat_budget(alphabet_bits=16), 59
        )

    def test_bfloat16_rounding_is_not_associative(self) -> None:
        half_ulp = 2.0**-8
        self.assertEqual(round_to_bfloat16(1.0 + half_ulp), 1.0)
        self.assertEqual(bf16_add(half_ulp, half_ulp), 2.0**-7)
        result = native_rounding_semiring_counterexample()
        self.assertEqual(result["left_associated"], 1.0)
        self.assertEqual(result["right_associated"], 1.0 + 2.0**-7)
        self.assertFalse(result["addition_is_associative"])
        self.assertFalse(
            result["finite_semiring_premise_holds_for_native_bf16_addition"]
        )

    def test_fp32_accumulation_is_not_associative(self) -> None:
        self.assertEqual(fp32_add(2.0**24, 1.0), 2.0**24)
        result = native_fp32_accumulation_counterexample()
        self.assertEqual(result["left_associated"], 0.0)
        self.assertEqual(result["right_associated"], 1.0)
        self.assertFalse(result["addition_is_associative"])
        self.assertFalse(
            result[
                "finite_semiring_premise_holds_for_native_fp32_addition"
            ]
        )

    def test_audit_rejects_constructor_without_closing_general_gap(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertTrue(boundary["published_boolean_or_finite_semiring_algorithm"])
        self.assertTrue(boundary["published_explicit_adjacency_catalog_charged"])
        self.assertEqual(
            boundary["compressed_or_implicit_edge_catalog"], "NOT_COVERED"
        )
        self.assertFalse(boundary["q4_query_payload_fits_2_5_percent"])
        self.assertFalse(boundary["bf16_query_payload_fits_2_5_percent"])
        self.assertFalse(boundary["boolean_sidecar_fits_8_gib"])
        self.assertFalse(boundary["native_bf16_is_published_semiring"])
        self.assertFalse(
            boundary["native_fp32_accumulation_is_published_semiring"]
        )
        self.assertFalse(
            boundary["direct_32_query_no_reuse_boolean_fits_2_5_percent"]
        )
        self.assertEqual(boundary["general_nonlinear_numerical_rank_one_gap"], "OPEN")
        self.assertEqual(boundary["core_candidate"], "NONE")

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            williams_graph_layout(
                dimension=0, alphabet_bits=1, block_symbols=1
            )
        with self.assertRaises(ValueError):
            ideal_model_wide_substitution(
                alphabet_bits=0, block_symbols=1
            )
        with self.assertRaises(ValueError):
            minimum_block_symbols_for_fraction(Fraction(0, 1))
        with self.assertRaises(ValueError):
            minimum_block_symbols_for_dfloat_budget(alphabet_bits=0)
        with self.assertRaises(ValueError):
            round_to_bfloat16(float("inf"))


if __name__ == "__main__":
    unittest.main()
