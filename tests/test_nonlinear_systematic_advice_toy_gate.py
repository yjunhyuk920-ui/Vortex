from __future__ import annotations

import unittest

from vortex_runtime.nonlinear_systematic_advice_toy_gate import (
    DECISION,
    derive_audit,
    query_advice_cubes,
    rank_one_masks,
)


class NonlinearSystematicAdviceToyGateTests(unittest.TestCase):
    def test_rank_one_query_population_is_complete(self) -> None:
        masks = rank_one_masks()
        self.assertEqual(len(masks), 21)
        self.assertEqual(sum(mask.bit_count() > 2 for mask in masks), 6)

    def test_every_hard_query_has_fourteen_symbolic_advice_cubes(self) -> None:
        for mask in rank_one_masks():
            if mask.bit_count() > 2:
                self.assertEqual(len(query_advice_cubes(mask)), 14)

    def test_no_single_advice_truth_table_serves_all_queries(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        self.assertFalse(result["scheme_exists"])
        self.assertTrue(result["all_depth_two_decoder_trees_enumerated"])
        self.assertFalse(
            result["claim_boundary"][
                "fully_nonsystematic_seven_bit_encoding_covered"
            ]
        )
        self.assertFalse(
            result["claim_boundary"]["surviving_runtime_candidate"]
        )


if __name__ == "__main__":
    unittest.main()
