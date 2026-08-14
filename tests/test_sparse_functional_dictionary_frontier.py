from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import unittest

from vortex_runtime.sparse_functional_dictionary_frontier import (
    DECISION,
    DICTIONARY_FORM_BITS,
    REGISTERED_PARAMETERS,
    WORD_BITS,
    derive_audit,
    direct_square_table_case,
    largest_cardinality_ruled_out_probe_count,
    rank_one_tuple_cardinality_boundary,
    sparse_choice_log2_upper,
)


class SparseFunctionalDictionaryFrontierTests(unittest.TestCase):
    def test_rank_one_tuple_floor_is_tight_to_one_bit(self) -> None:
        result = rank_one_tuple_cardinality_boundary()
        self.assertEqual(result["matrix_count"], 884)
        self.assertEqual(result["minimum_matrix_side"], 1_024)
        self.assertEqual(
            result["parameters_reconstructed_from_shapes"],
            REGISTERED_PARAMETERS,
        )
        self.assertEqual(
            result["pair_description_bits_upper_bound"], 39_254_528
        )
        self.assertEqual(
            result["exact_floor_log2_query_count"], 39_254_527
        )

    def test_bit_form_cardinality_boundary_is_finite(self) -> None:
        floor = 39_254_527
        ruled_out = largest_cardinality_ruled_out_probe_count(
            items=DICTIONARY_FORM_BITS,
            strict_query_lower_exponent=floor,
        )
        self.assertEqual(ruled_out, 2_020_681)
        self.assertLessEqual(
            sparse_choice_log2_upper(
                items=DICTIONARY_FORM_BITS, probes=ruled_out
            ),
            Decimal(floor),
        )
        self.assertGreater(
            sparse_choice_log2_upper(
                items=DICTIONARY_FORM_BITS, probes=ruled_out + 1
            ),
            Decimal(floor),
        )

    def test_word_boundary_grants_every_internal_linear_choice(self) -> None:
        words = DICTIONARY_FORM_BITS // WORD_BITS
        ruled_out = largest_cardinality_ruled_out_probe_count(
            items=words,
            strict_query_lower_exponent=39_254_527,
            choice_bits_per_item=WORD_BITS,
        )
        self.assertEqual(ruled_out, 494_025)

    def test_direct_target_fraction_table_has_fatal_space(self) -> None:
        result = direct_square_table_case(10)
        self.assertEqual(result["raw_cells_per_tile"], 100)
        self.assertEqual(
            result["stored_nonzero_rank_one_forms_per_tile"], 1_046_529
        )
        self.assertEqual(
            Fraction(result["storage_ratio_over_one_bit_source"]),
            Fraction(1_046_529, 100),
        )
        self.assertEqual(
            Fraction(result["worst_query_fraction"]), Fraction(1, 100)
        )
        self.assertGreater(result["favorable_persistent_tib"], 400)

    def test_audit_keeps_dictionary_open_but_admits_no_component(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        counting = result["near_linear_sparse_dictionary_counting"]
        bit_case = counting["bit_forms"]
        word_case = counting["word_64_favorable"]
        self.assertEqual(
            bit_case["capacity_only_witness_probe_count"], 2_216_796
        )
        self.assertEqual(
            word_case["minimum_probe_count_not_ruled_out_by_cardinality"],
            494_026,
        )
        self.assertEqual(
            word_case["capacity_only_witness_probe_count"], 510_961
        )
        self.assertEqual(
            word_case["minimum_not_ruled_out_payload_bytes"], 3_952_208
        )
        self.assertTrue(
            bit_case[
                "random_dictionary_bad_dependency_union_below_data_bits"
            ]
        )
        self.assertTrue(
            bit_case[
                "full_rank_high_girth_dictionary_exists_by_union_bound"
            ]
        )
        self.assertLess(
            bit_case[
                "random_dictionary_rank_deficiency_probability_log2_upper"
            ],
            0,
        )
        self.assertFalse(bit_case["capacity_is_aligned_to_rank_one_queries"])
        boundary = result["claim_boundary"]
        self.assertFalse(boundary["explicit_rank_one_aligned_dictionary"])
        self.assertFalse(boundary["native_numerical_lift"])
        self.assertFalse(boundary["target_candidate"])
        self.assertEqual(boundary["model_forward_calls"], 0)
        self.assertEqual(boundary["hardware_actions"], 0)
        self.assertIn("NO_SURVIVING_CANDIDATE", result["decisions"])

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            direct_square_table_case(0)
        with self.assertRaises(ValueError):
            sparse_choice_log2_upper(items=10, probes=6)
        with self.assertRaises(ValueError):
            largest_cardinality_ruled_out_probe_count(
                items=10, strict_query_lower_exponent=0
            )


if __name__ == "__main__":
    unittest.main()
