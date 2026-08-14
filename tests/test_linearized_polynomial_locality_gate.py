from __future__ import annotations

import unittest

from vortex_runtime.linearized_polynomial_locality_gate import (
    DECISION,
    dense_ordinary_polynomial_data_structure_screen,
    derive_audit,
    full_coefficient_sweep_floor,
    linearized_representation_accounting,
)


class LinearizedPolynomialLocalityGateTests(unittest.TestCase):
    def test_representation_is_information_preserving_not_compressing(self) -> None:
        result = linearized_representation_accounting(16_384)
        self.assertEqual(result["linearized_coefficients"], 16_384)
        self.assertEqual(result["base_bits_per_coefficient"], 16_384)
        self.assertEqual(result["total_base_bits"], 16_384**2)
        self.assertEqual(result["total_base_bits"], result["matrix_base_bits"])
        self.assertTrue(result["evaluation_map_bijective"])
        self.assertFalse(result["representation_compresses_arbitrary_matrix"])

    def test_even_one_bit_full_sweep_fails_registered_latency(self) -> None:
        result = full_coefficient_sweep_floor()
        self.assertEqual(result["one_bit_model_bytes"], "50731155456")
        self.assertEqual(
            result["one_bit_milliseconds_per_token"], "49.542144000"
        )
        self.assertEqual(result["one_bit_target_multiple"], "2.4771072")
        self.assertFalse(result["passes_target"])

    def test_compressed_checkpoint_full_sweep_is_farther_out(self) -> None:
        result = full_coefficient_sweep_floor()
        self.assertEqual(
            result["compressed_milliseconds_per_token"], "538.30078125000"
        )
        self.assertEqual(
            result["compressed_target_multiple"], "26.9150390625"
        )

    def test_dense_polynomial_ds_sees_exponential_ordinary_degree(self) -> None:
        result = dense_ordinary_polynomial_data_structure_screen()
        self.assertEqual(result["ordinary_degree"], "2^16383")
        self.assertEqual(result["dense_base_bits_log2_strict_lower"], 16_397)
        self.assertEqual(result["source_base_bits_log2"], 28)
        self.assertTrue(result["dense_ds_import_rejected"])

    def test_audit_changes_mechanism_without_overclaiming(self) -> None:
        result = derive_audit()
        self.assertEqual(result["decision"], DECISION)
        boundary = result["claim_boundary"]
        self.assertFalse(boundary["all_preprocessed_linearized_polynomial_ds_rejected"])
        self.assertFalse(boundary["native_bf16_fp32_lift"])
        self.assertFalse(boundary["target_candidate"])
        self.assertIn("CHANGE_MECHANISM_CLASS", result["decisions"])

    def test_invalid_shapes_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            linearized_representation_accounting(0)
        with self.assertRaises(ValueError):
            dense_ordinary_polynomial_data_structure_screen(12)


if __name__ == "__main__":
    unittest.main()
