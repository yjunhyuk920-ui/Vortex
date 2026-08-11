from __future__ import annotations

import unittest

from vortex_runtime.extension_field_rank_saturating_frontier import (
    DECISION,
    REGISTERED_PARAMETERS,
    aggregate_full_ambient_materialization,
    derive_audit,
    exhaustive_two_by_two_best,
    full_ambient_rank_one_materialization,
    query_specialized_identity_batch,
    small_dictionary_witness,
)


class ExtensionFieldRankSaturatingFrontierTests(unittest.TestCase):
    def test_square_full_ambient_case_has_exact_published_dimension(self) -> None:
        result = full_ambient_rank_one_materialization(16_384, 16_384)
        self.assertEqual(
            result["q_system_binary_basis_dimension_rho_1"],
            268_419_073,
        )
        self.assertEqual(result["direct_field_summary_bits"], 4_397_778_092_032)
        self.assertAlmostEqual(
            result["direct_summary_to_raw_ratio_decimal"],
            16_383.000061035156,
        )

    def test_model_aggregate_reconstructs_registered_population(self) -> None:
        result = aggregate_full_ambient_materialization()
        self.assertEqual(result["raw_one_bit_source_bits"], REGISTERED_PARAMETERS)
        self.assertEqual(
            result["direct_field_summary_bits"], 6_584_324_197_576_704
        )
        self.assertAlmostEqual(result["direct_summary_tib"], 748.5509965564124)

    def test_query_specialized_identity_fails_even_favorable_io(self) -> None:
        result = query_specialized_identity_batch(32)
        self.assertEqual(
            result["uniform_batch_union_probability"],
            "4294967295/4294967296",
        )
        self.assertEqual(result["expected_active_cold_bytes_ceiling"], 42_141_220_855)
        self.assertGreater(result["allowance_ratio"], 3)
        self.assertGreater(
            result["zero_compute_ms_per_token_at_favorable_bandwidth"], 40
        )

    def test_two_by_two_optimum_is_exactly_two_atoms(self) -> None:
        result = exhaustive_two_by_two_best(5)
        self.assertEqual(result["full_rank_dictionaries_tested"], 2_688)
        self.assertEqual(result["maximum_minimum_atoms"], 2)
        self.assertEqual(result["distance_histogram"], {"0": 1, "1": 5, "2": 4})

    def test_explicit_two_by_three_witness_has_nearly_full_batch_union(self) -> None:
        result = small_dictionary_witness(
            2, 3, (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x3F)
        )
        self.assertEqual(result["maximum_minimum_atoms"], 3)
        self.assertGreater(result["canonical_expected_union_fraction"], 0.999)
        self.assertFalse(result["is_general_lower_bound"])

    def test_audit_keeps_nonlinear_decoder_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertFalse(boundary["published_rho_is_number_of_probed_cells"])
        self.assertFalse(boundary["nonlinear_minimum_weight_decoder_rejected"])
        self.assertFalse(boundary["target_candidate"])
        self.assertEqual(boundary["model_forward_calls"], 0)
        self.assertIn("NO_SURVIVING_CANDIDATE", result["decisions"])

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            full_ambient_rank_one_materialization(0, 2)
        with self.assertRaises(ValueError):
            query_specialized_identity_batch(0)
        with self.assertRaises(ValueError):
            exhaustive_two_by_two_best(3)


if __name__ == "__main__":
    unittest.main()
