from __future__ import annotations

import unittest

from vortex_runtime.cartesian_bilinear_causal_utility_gate import (
    DECISION,
    cartesian_bilinear,
    causal_utility_accounting,
    derive_audit,
    full_sweep_accounting,
)


class CartesianBilinearCausalUtilityGateTests(unittest.TestCase):
    def test_cartesian_identity_returns_every_pair(self) -> None:
        weight = [[1, 0, 1], [0, 1, 1]]
        left = [[1, 0], [0, 1]]
        states = [[1, 0], [1, 1], [0, 1]]
        self.assertEqual(
            cartesian_bilinear(left, weight, states), [[1, 1], [1, 0]]
        )

    def test_scalar_count_does_not_create_states(self) -> None:
        result = causal_utility_accounting(32, 32)
        self.assertEqual(result["cartesian_exact_scalar_count"], 1024)
        self.assertEqual(result["maximum_distinct_forward_states"], 32)
        self.assertEqual(result["maximum_causal_tokens_from_these_states"], 32)
        self.assertFalse(result["cartesian_scalars_may_be_counted_as_tokens"])

    def test_optimistic_full_sweep_still_fails(self) -> None:
        result = full_sweep_accounting()
        self.assertEqual(result["optimistic_sweep_bytes"], "537750247833.6")
        self.assertEqual(result["optimistic_sweep_seconds"], "16.8046952448")
        self.assertEqual(result["valid_milliseconds_per_token"], "525.1467264000")
        self.assertEqual(result["valid_target_multiple"], "26.25733632")
        self.assertEqual(
            result["minimum_causally_usable_tokens_per_full_sweep"], 841
        )
        self.assertFalse(result["registered_states_meet_full_sweep_floor"])

    def test_false_k_squared_denominator_is_detected(self) -> None:
        result = full_sweep_accounting()
        self.assertEqual(
            result["invalid_scalar_denominator_milliseconds"],
            "16.4108352000",
        )
        self.assertTrue(result["invalid_scalar_denominator_would_pass"])

    def test_registered_dfloat_floor_is_unchanged(self) -> None:
        result = full_sweep_accounting()
        self.assertEqual(
            result["dfloat_milliseconds_per_token"], "538.30078125000"
        )

    def test_audit_keeps_partial_sources_open(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertFalse(boundary["partial_evidence_source_rejected"])
        self.assertFalse(boundary["nonlinear_rank_one_source_rejected"])
        self.assertFalse(boundary["full_sweep_core_passes"])

    def test_invalid_counts_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            causal_utility_accounting(0, 32)
        with self.assertRaises(ValueError):
            full_sweep_accounting(0)


if __name__ == "__main__":
    unittest.main()
