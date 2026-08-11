from __future__ import annotations

import unittest

import numpy as np

from vortex_runtime.native_exact_shortcut_frontier import (
    DECISION,
    REQUIRED_ELIMINATION_FRACTION,
    derive_audit,
    exact_product_reuse_metrics,
    exact_value_multiplicity_metrics,
    favorable_rounding_absorption_upper,
    roundlock_oracle_metrics,
    temporal_identity_metrics,
)


class NativeExactShortcutFrontierTests(unittest.TestCase):
    def test_temporal_metrics_count_exact_words(self) -> None:
        inputs = np.asarray(
            [[1, 2, 2, 3], [1, 4, 2, 3], [5, 4, 6, 3]],
            dtype=np.float32,
        )
        result = temporal_identity_metrics(inputs)
        self.assertEqual(result["shape"], [3, 4])
        self.assertEqual(
            result["consecutive_same_coordinate_fraction"]["min"], 0.5
        )
        self.assertEqual(result["last_history_hit_fraction"], 0.5)
        self.assertEqual(result["natural_run_fraction"]["min"], 0.75)

    def test_product_metrics_find_duplicates_and_opposites(self) -> None:
        weight = np.asarray([[1, 1, -1, 2]], dtype=np.float32)
        vector = np.ones(4, dtype=np.float32)
        result = exact_product_reuse_metrics(weight, vector)
        self.assertEqual(result["unique_product_fraction"]["min"], 0.75)
        self.assertEqual(result["best_duplicate_reuse_fraction"], 0.25)
        self.assertEqual(
            result["perfect_opposite_pair_cancellation_fraction"]["max"],
            0.5,
        )

    def test_value_multiplicity_checks_both_orientations(self) -> None:
        words = np.asarray([[1, 1, 2], [1, 3, 2]], dtype=np.uint16)
        result = exact_value_multiplicity_metrics(words)
        self.assertEqual(result["unique_values_per_column"]["min"], 1.0)
        self.assertEqual(result["unique_values_per_row"]["min"], 2.0)
        self.assertEqual(result["best_possible_d1_forward_work_ratio"], 0.5)

    def test_rounding_screen_is_bounded(self) -> None:
        weight = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
        vector = np.asarray([1, 1], dtype=np.float32)
        result = favorable_rounding_absorption_upper(weight, vector)
        self.assertGreaterEqual(result["min"], 0.0)
        self.assertLessEqual(result["max"], 1.0)

    def test_roundlock_counts_bf16_words_and_fails_closed(self) -> None:
        candidate = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        target = np.asarray([1.0, 2.5, 3.0], dtype=np.float32)
        result = roundlock_oracle_metrics(candidate, target)
        self.assertEqual(result["locked_coordinates"], 2)
        self.assertEqual(result["oracle_lock_fraction"], 2 / 3)
        self.assertFalse(result["whole_vector_locked"])
        with self.assertRaises(ValueError):
            roundlock_oracle_metrics(candidate, target[:2])

    def test_audit_uses_no_old_two_point_five_assumption(self) -> None:
        weight = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
        words = np.asarray([[1, 2], [3, 4]], dtype=np.uint16)
        prefix = np.asarray([[1, 2], [2, 3]], dtype=np.float32)
        result = derive_audit(
            weight=weight,
            weight_words=words,
            prefix_inputs=prefix,
            current_input=prefix[-1],
            roundlock_pairs={"atlas_plus_one_page_down": (prefix[0], prefix[1])},
        )
        self.assertEqual(result["decision"], DECISION)
        self.assertFalse(
            result["registered_budget"]["old_2_5_percent_assumption_used"]
        )
        self.assertGreater(REQUIRED_ELIMINATION_FRACTION, 0.98)
        self.assertFalse(result["claim_boundary"]["target_achieved"])
        self.assertIn("omega_roundlock_atlas_one_page", result["candidate_decisions"])

    def test_invalid_shapes_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            temporal_identity_metrics(np.ones((1, 2), dtype=np.float32))
        with self.assertRaises(ValueError):
            exact_product_reuse_metrics(
                np.ones((2, 2), dtype=np.float32),
                np.ones(3, dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
