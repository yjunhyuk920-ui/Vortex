from __future__ import annotations

import unittest

from vortex_runtime.boolean_oracle_parity_lift_gate import (
    adversarial_certificate,
    audit_payload,
    boolean_row_or,
    canonical_isolating_queries,
    exhaustive_short_transcript_check,
    exhaustive_maximum_one_rectangle,
    exact_inner_product_boolean_rank,
    gf2_row_parity,
    inner_product_one_entries,
    indistinguishable_deletions,
    maximum_inner_product_one_rectangle_entries,
    minimum_black_box_boolean_calls,
    one_shot_boolean_feature_lift_projection,
    query_isolates_coordinate,
    support_mask,
)


class BooleanOracleParityLiftGateTests(unittest.TestCase):
    def test_one_deletion_flips_target_parity(self) -> None:
        for width in range(1, 9):
            target = (1 << width) - 1
            row0 = target
            base = gf2_row_parity(row0, target)
            for coordinate in range(width):
                row_i = row0 & ~(1 << coordinate)
                self.assertNotEqual(base, gf2_row_parity(row_i, target))

    def test_only_singleton_distinguishes_full_support_deletion(self) -> None:
        width = 6
        target = (1 << width) - 1
        for coordinate in range(width):
            row0 = target
            row_i = row0 & ~(1 << coordinate)
            for query in range(1 << width):
                differs = boolean_row_or(row0, query) != boolean_row_or(row_i, query)
                self.assertEqual(differs, query == (1 << coordinate))
                self.assertEqual(
                    query_isolates_coordinate(query, target, coordinate),
                    query == (1 << coordinate),
                )

    def test_partial_support_query_may_include_irrelevant_coordinates(self) -> None:
        width = 7
        target = support_mask(width, [1, 3, 5])
        query = support_mask(width, [0, 3, 6])
        self.assertTrue(query_isolates_coordinate(query, target, 3))
        self.assertFalse(query_isolates_coordinate(query, target, 1))

    def test_fewer_than_support_size_queries_leave_opposite_parity_witness(self) -> None:
        width = 8
        target = (1 << width) - 1
        queries = tuple(1 << coordinate for coordinate in range(width - 1))
        certificate = adversarial_certificate(queries, target)
        self.assertFalse(certificate["exact_parity_determined_on_adversary_family"])
        self.assertEqual(certificate["indistinguishable_deletions"], [width - 1])
        witness = certificate["opposite_parity_witness"]
        self.assertTrue(witness["transcript_equal"])
        self.assertNotEqual(witness["row0_parity"], witness["deletion_parity"])

    def test_canonical_support_isolation_is_tight(self) -> None:
        width = 9
        target = support_mask(width, [0, 2, 4, 8])
        queries = canonical_isolating_queries(width, target)
        self.assertEqual(len(queries), 4)
        self.assertEqual(indistinguishable_deletions(queries, target), ())
        self.assertEqual(minimum_black_box_boolean_calls(4), 4)

    def test_exhaustive_short_transcripts(self) -> None:
        for width in range(1, 6):
            result = exhaustive_short_transcript_check(width)
            self.assertEqual(result["false_short_determinations"], 0)
            self.assertTrue(result["canonical_determines_family"])
            self.assertEqual(result["canonical_queries"], width)

    def test_exact_boolean_rank_closed_form(self) -> None:
        for width in range(1, 9):
            self.assertEqual(
                inner_product_one_entries(width),
                ((1 << width) - 1) * (1 << (width - 1)),
            )
            self.assertEqual(
                maximum_inner_product_one_rectangle_entries(width),
                1 << (width - 1),
            )
            self.assertEqual(exact_inner_product_boolean_rank(width), (1 << width) - 1)

    def test_small_rectangle_bound_is_exhaustively_tight(self) -> None:
        for width in range(1, 5):
            self.assertEqual(
                exhaustive_maximum_one_rectangle(width),
                maximum_inner_product_one_rectangle_entries(width),
            )

    def test_one_shot_feature_lift_is_exponential(self) -> None:
        projection = one_shot_boolean_feature_lift_projection(25, 25)
        self.assertEqual(projection["minimum_boolean_features"], (1 << 25) - 1)
        self.assertGreater(projection["feature_storage_over_source"], 1_000_000)

    def test_audit_keeps_non_black_box_routes_open(self) -> None:
        payload = audit_payload(widths=(8, 216), word_bits=64)
        self.assertEqual(
            payload["decision"],
            "REJECT_BLACK_BOX_BOOLEAN_TO_F2_LIFT_AS_SUBDENSE_CORE",
        )
        self.assertEqual(
            payload["claim_boundary"]["direct_gf2_data_structure"], "OPEN"
        )
        self.assertEqual(
            payload["claim_boundary"]["single_transformed_boolean_product_feature_lift"],
            "REJECTED_EXPONENTIAL_FEATURES",
        )
        self.assertEqual(
            payload["claim_boundary"]["target_hardware"], "NOT_TESTED"
        )
        projection = payload["width_projections"][1]
        self.assertEqual(projection["required_boolean_product_calls"], 216)
        self.assertEqual(projection["complete_response_over_source_bits"], 1.0)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            minimum_black_box_boolean_calls(0)
        with self.assertRaises(ValueError):
            support_mask(4, [4])
        with self.assertRaises(ValueError):
            exhaustive_short_transcript_check(6)


if __name__ == "__main__":
    unittest.main()
