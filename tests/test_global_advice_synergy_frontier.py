from __future__ import annotations

from fractions import Fraction
import unittest

from vortex_runtime.global_advice_synergy_frontier import (
    DECISION,
    derive_audit,
    exhaustive_xor_synergy_control,
    favorable_ckl_tile_sum_screen,
    recover_from_xor_advice,
    registered_full_hidden_square_tiles,
    registered_square_tile_average_screen,
    separate_tile_injectivity_probe_floor,
    xor_advice,
)


class GlobalAdviceSynergyFrontierTests(unittest.TestCase):
    def test_registered_full_square_tile_count(self) -> None:
        result = registered_full_hidden_square_tiles()
        self.assertEqual(result["full_tiles_per_gate_up_down_matrix"], 3)
        self.assertEqual(result["full_tiles_per_layer"], 11)
        self.assertEqual(result["full_tiles_total"], 1_386)

    def test_xor_advice_recovers_every_target_given_all_others(self) -> None:
        words = (0b001, 0b110, 0b101, 0b011)
        advice = xor_advice(words, symbol_bits=3)
        for target in range(len(words)):
            known = words[:target] + words[target + 1 :]
            self.assertEqual(
                recover_from_xor_advice(
                    advice, known, symbol_bits=3
                ),
                words[target],
            )

    def test_exhaustive_synergy_exceeds_advice_entropy_sum(self) -> None:
        result = exhaustive_xor_synergy_control(
            instances=4, symbol_bits=3
        )
        self.assertEqual(result["cases"], 4_096)
        self.assertEqual(result["recovery_checks"], 16_384)
        self.assertTrue(result["advice_is_uniform"])
        self.assertEqual(result["advice_entropy_bits"], 3)
        self.assertEqual(result["sum_conditional_information_bits"], 12)
        self.assertEqual(
            result["conditional_information_over_advice_entropy"], 4
        )

    def test_invalid_average_happens_to_enter_displayed_single_tile_range(
        self,
    ) -> None:
        result = registered_square_tile_average_screen()
        self.assertEqual(result["full_tiles_total"], 1_386)
        average = Fraction(result["invalid_even_average_redundancy_bits"])
        self.assertGreater(average, 4_194_304)
        self.assertLessEqual(average, 67_108_864)
        self.assertTrue(
            result["average_inside_theorem_statement_upper_range"]
        )
        self.assertFalse(result["average_inside_displayed_proof_regime"])
        self.assertTrue(
            result["hidden_asymptotic_constant_set_to_one_counterfactually"]
        )
        self.assertFalse(
            result["leading_monomial_is_certified_finite_lower_bound"]
        )
        self.assertFalse(result["direct_sum_theorem_supplied"])

    def test_separate_tile_injectivity_floor_is_finite_and_scoped(self) -> None:
        self.assertEqual(
            separate_tile_injectivity_probe_floor(
                tile_bits=36, local_advice_cap_bits=7
            ),
            1,
        )
        self.assertEqual(
            separate_tile_injectivity_probe_floor(
                tile_bits=36, local_advice_cap_bits=36
            ),
            0,
        )

    def test_even_invalid_sum_of_separate_floors_is_below_target(self) -> None:
        result = favorable_ckl_tile_sum_screen()
        self.assertEqual(
            result["minimum_dimension_with_average_r_at_least_n"], 6
        )
        self.assertEqual(
            result["favorable_global_reshape_tile_count"], 11_215_219_370
        )
        self.assertFalse(
            result["even_invalid_sum_of_separate_floors_reaches_target"]
        )
        self.assertEqual(result["granted_ceiling_local_advice_cap_bits"], 7)
        self.assertTrue(
            result["average_redundancy_inside_theorem_statement_range"]
        )
        self.assertEqual(result["separate_tile_injectivity_probe_floor"], 1)
        self.assertTrue(result["leading_monomial_uses_exact_invalid_average"])
        self.assertFalse(result["simultaneous_hard_query_composition_proved"])
        self.assertFalse(
            result["average_redundancy_inside_displayed_proof_regime"]
        )
        self.assertGreater(
            result["requested_block_over_invalid_floor_sum"], 9.8
        )
        self.assertLess(
            result[
                "counterfactual_displayed_proof_coefficient_outside_regime"
            ],
            1,
        )

    def test_audit_rejects_only_the_naive_lift(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertFalse(boundary["naive_advice_division_is_valid"])
        self.assertTrue(boundary["single_matrix_ckl_theorem_is_valid"])
        self.assertFalse(boundary["published_cross_matrix_direct_sum"])
        self.assertFalse(boundary["target_scale_lower_bound_established"])
        self.assertEqual(boundary["general_nonlinear_rank_one_gap"], "OPEN")

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            xor_advice((1,), symbol_bits=1)
        with self.assertRaises(ValueError):
            xor_advice((0, 2), symbol_bits=1)
        with self.assertRaises(ValueError):
            exhaustive_xor_synergy_control(instances=1, symbol_bits=1)
        with self.assertRaises(ValueError):
            separate_tile_injectivity_probe_floor(
                tile_bits=0, local_advice_cap_bits=0
            )


if __name__ == "__main__":
    unittest.main()
