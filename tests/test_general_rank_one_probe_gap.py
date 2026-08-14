from __future__ import annotations

from fractions import Fraction
import unittest

from vortex_runtime.general_rank_one_probe_gap import (
    DECISION,
    DFLOAT11_REPORTED_BITS,
    GLOBAL_ADVICE_BITS,
    LARGEST_REGISTERED_SQUARE_DIMENSION,
    LLAMA_405B_PARAMETERS,
    Q4_BITS,
    ckl_single_matrix_boundary,
    derive_audit,
    ko_theorem_substitution,
    nrs_help_bit_substitution,
    rank_one_query_log2_upper_bound,
)


class GeneralRankOneProbeGapTests(unittest.TestCase):
    def test_nrs_help_bits_are_general_but_target_scale_tiny(self) -> None:
        one_plane = nrs_help_bit_substitution(LLAMA_405B_PARAMETERS)
        q4 = nrs_help_bit_substitution(Q4_BITS)
        dfloat = nrs_help_bit_substitution(DFLOAT11_REPORTED_BITS)
        self.assertEqual(one_plane["parity_bits_per_instance"], 5)
        self.assertEqual(q4["parity_bits_per_instance"], 23)
        self.assertEqual(dfloat["parity_bits_per_instance"], 64)
        self.assertEqual(
            dfloat["forced_64_bit_word_probes_for_one_instance"], 1
        )
        self.assertTrue(dfloat["covers_arbitrary_nonlinear_help_bits"])
        self.assertTrue(dfloat["covers_cross_instance_input_probes"])
        self.assertFalse(dfloat["target_scale_rejection"])
        self.assertGreater(
            dfloat["requested_budget_to_forced_bit_probe_ratio"],
            1_000_000_000,
        )

    def test_rank_one_query_count_is_only_exponential_in_dimension(self) -> None:
        self.assertEqual(rank_one_query_log2_upper_bound(16_384), 32_768)
        self.assertEqual(
            rank_one_query_log2_upper_bound(16_384, field_symbol_bits=16),
            524_288,
        )

    def test_ko_finite_premises_do_not_reach_two_point_five_percent(
        self,
    ) -> None:
        dimension = LARGEST_REGISTERED_SQUARE_DIMENSION
        data_bits = dimension * dimension
        result = ko_theorem_substitution(
            data_bits=data_bits,
            query_collection_log2_upper_bound=(2 * dimension),
            target_probe_bits=data_bits // 40,
        )
        self.assertEqual(
            result[
                "maximum_integer_t_allowed_by_collection_entropy_premise"
            ],
            0,
        )
        self.assertFalse(result["entropy_premise_holds_at_target_t"])
        self.assertFalse(
            result[
                "query_universe_can_meet_even_non_omega_count_at_target_t"
            ]
        )
        self.assertFalse(result["finite_omega_threshold_supplied_by_theorem"])
        self.assertFalse(result["target_scale_rejection"])

    def test_ckl_single_matrix_range_cannot_absorb_global_advice(self) -> None:
        result = ckl_single_matrix_boundary()
        self.assertEqual(result["matrix_bits"], 268_435_456)
        self.assertEqual(
            result["published_redundancy_range_upper_bits"], 67_108_864
        )
        self.assertEqual(result["global_advice_bits"], GLOBAL_ADVICE_BITS)
        self.assertEqual(result["global_advice_over_range_ceiling"], "1024")
        self.assertFalse(
            result["single_matrix_theorem_applicable_with_global_advice"]
        )
        self.assertFalse(result["model_wide_direct_sum_supplied"])

    def test_registered_audit_keeps_both_universal_claims_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        self.assertEqual(
            result["frozen_model"]["metadata_kv_state_scheduler_reprop_cost"],
            "ZERO_FAVORABLE_GRANT",
        )
        target = result["registered_target"]
        self.assertEqual(target["requested_block_bits"], "110244000000")
        self.assertEqual(
            target["requested_per_certified_token_bytes"], "430640625"
        )
        self.assertEqual(
            Fraction(target["global_advice_fraction_of_dfloat_bits"]),
            Fraction(GLOBAL_ADVICE_BITS, DFLOAT11_REPORTED_BITS),
        )
        self.assertGreater(
            result["ko_2025"][
                "requested_budget_over_all_linear_t_ceiling"
            ],
            10_000,
        )
        ko = result["ko_2025"]
        self.assertIn(
            "CHARGES_PROBES_TO_ALL_PREPROCESSED_S_CELLS",
            ko["probe_accounting_mismatch"],
        )
        self.assertEqual(
            ko["free_global_advice_checkpoint_probe_bound"], "NOT SUPPLIED"
        )
        nrs = result["nrs_help_bits"]
        self.assertIn(
            "DOES_NOT_EXPOSE_FOUR_INDEPENDENT_PARITIES",
            nrs["q4_native_query_caveat"],
        )
        self.assertIn(
            "NOT_PROVEN_INDEPENDENT_PARITY_INPUTS",
            nrs["dfloat_independence_caveat"],
        )
        boundary = result["claim_boundary"]
        self.assertEqual(
            boundary["general_nonlinear_rank_one_constructor"],
            "NOT FOUND",
        )
        self.assertEqual(
            boundary["general_nonlinear_rank_one_target_lower_bound"],
            "NOT PROVED",
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

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            nrs_help_bit_substitution(0)
        with self.assertRaises(ValueError):
            rank_one_query_log2_upper_bound(0)
        with self.assertRaises(ValueError):
            ko_theorem_substitution(
                data_bits=0,
                query_collection_log2_upper_bound=1,
                target_probe_bits=1,
            )


if __name__ == "__main__":
    unittest.main()
