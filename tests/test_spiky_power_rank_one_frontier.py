from __future__ import annotations

from math import comb
import unittest

import numpy as np

from vortex_runtime.spiky_power_rank_one_frontier import (
    ALLOWED_WORK_FRACTION,
    DECISION,
    derive_audit,
    model_wide_spiky_description_gate,
    powerfold_expansion_gate,
    spiky_component_scalar,
    spiky_direct_description_gate,
)


class SpikyPowerRankOneFrontierTests(unittest.TestCase):
    def test_spiky_scalar_equation_matches_explicit_matrix(self) -> None:
        left = np.asarray([2.0, -1.0, 3.0])
        right = np.asarray([4.0, 5.0, -2.0, 1.0])
        rows = np.asarray([1.5, 2.0, -1.0])
        columns = np.asarray([2.0, -3.0, 0.5, 4.0])
        row_labels = np.asarray([1, 2, 1])
        column_labels = np.asarray([2, 1, 0, 2])

        mask = (
            (row_labels[:, None] == column_labels[None, :])
            & (row_labels[:, None] > 0)
        )
        matrix = mask * rows[:, None] * columns[None, :]
        expected = float(left @ matrix @ right)
        observed = spiky_component_scalar(
            left,
            right,
            rows,
            columns,
            row_labels,
            column_labels,
        )
        self.assertAlmostEqual(observed, expected)

    def test_registered_spiky_description_gate_is_decisive(self) -> None:
        result = spiky_direct_description_gate()
        self.assertEqual(result["dimension"], 16_384)
        self.assertEqual(result["allowed_factor_incidence_terms"], 3_174_800)
        self.assertTrue(result["hard_sign_matrix_exists"])
        self.assertGreater(result["hard_instance_exponent_bits"], 100_000_000)
        self.assertEqual(
            result["decision"],
            "REJECT_OMEGA_SPIKECUT_DIRECT_EVALUATOR_BY_FINITE_DESCRIPTION_GATE",
        )

    def test_model_wide_spiky_gate_grants_global_component_coupling(self) -> None:
        result = model_wide_spiky_description_gate()
        self.assertEqual(result["coefficient_count"], 403_747_897_344)
        self.assertEqual(result["matrix_instances"], 883)
        self.assertEqual(result["allowed_factor_incidence_terms"], 4_800_000_000)
        self.assertTrue(result["global_cross_matrix_components_granted"])
        self.assertTrue(result["invalid_cross_matrix_cells_ignored"])
        self.assertTrue(result["hard_sign_checkpoint_exists"])
        self.assertGreater(
            result["hard_checkpoint_exponent_bits"], 200_000_000_000
        )
        self.assertEqual(
            result["decision"],
            "REJECT_OMEGA_SPIKECUT_DIRECT_EVALUATOR_BY_MODEL_WIDE_DESCRIPTION_GATE",
        )

    def test_paper_dense_factor_diagnostic_misses_allowance(self) -> None:
        result = spiky_direct_description_gate()
        self.assertGreater(
            result["paper_dense_factor_diagnostic_fraction"],
            ALLOWED_WORK_FRACTION,
        )
        self.assertGreater(
            result["paper_dense_factor_diagnostic_over_allowance"], 1.0
        )

    def test_powerfold_thresholds_match_multinomial_terms(self) -> None:
        result = powerfold_expansion_gate()
        self.assertEqual(result["expanded_separable_term_cap"], 96)
        expected_ranks = {"2": 13, "3": 7, "4": 5, "5": 4}
        for power_text, root_rank in expected_ranks.items():
            power = int(power_text)
            row = result["power_thresholds"][power_text]
            self.assertEqual(
                row["maximum_root_rank_within_free_term_cap"], root_rank
            )
            self.assertEqual(
                row["expanded_terms_at_maximum"],
                comb(root_rank + power - 1, power),
            )
            self.assertGreater(row["expanded_terms_at_next_rank"], 96)

    def test_audit_keeps_general_gap_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        self.assertFalse(
            result["registered_budget"]["old_2_5_percent_assumption_used"]
        )
        self.assertFalse(result["claim_boundary"]["target_achieved"])
        self.assertFalse(
            result["claim_boundary"]["general_nonlinear_probe_gap_resolved"]
        )

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            spiky_direct_description_gate(dimension=4)
        with self.assertRaises(ValueError):
            spiky_direct_description_gate(allowed_work_fraction=1.0)
        with self.assertRaises(ValueError):
            powerfold_expansion_gate(powers=(1,))
        with self.assertRaises(ValueError):
            model_wide_spiky_description_gate(coefficient_count=0)
        with self.assertRaises(ValueError):
            spiky_component_scalar(
                np.ones(2),
                np.ones(2),
                np.ones(3),
                np.ones(2),
                np.ones(2, dtype=np.int64),
                np.ones(2, dtype=np.int64),
            )


if __name__ == "__main__":
    unittest.main()
