from __future__ import annotations

from fractions import Fraction
import itertools
import json
from pathlib import Path
import unittest

from vortex_runtime.average_oracle_amplifier_frontier import (
    DECISION,
    HOT_BITS,
    REGISTERED_BINARY_BITS,
    REGISTERED_OUTPUT_ROWS,
    derive_audit,
    fourier_list_size_upper_bound,
    minimum_self_contained_state_bits,
    parity_agreement_count,
    registered_population_from_records,
    self_contained_advantage_ceiling,
    two_call_self_correction_accuracy,
)


class AverageOracleAmplifierFrontierTests(unittest.TestCase):
    def test_fourier_list_bound_exact_arithmetic(self) -> None:
        self.assertEqual(fourier_list_size_upper_bound(Fraction(1, 4)), 4)
        self.assertEqual(fourier_list_size_upper_bound(Fraction(1, 100)), 2500)

    def test_exhaustive_small_truth_tables_obey_parseval_list_bound(self) -> None:
        dimension = 3
        advantage = Fraction(1, 8)
        bound = fourier_list_size_upper_bound(advantage)
        largest = 0
        for values in itertools.product((0, 1), repeat=1 << dimension):
            largest = max(
                largest,
                parity_agreement_count(
                    values,
                    dimension=dimension,
                    advantage=advantage,
                ),
            )
        self.assertLessEqual(largest, bound)

    def test_two_call_random_shift_corrects_a_close_linear_function(self) -> None:
        dimension = 3
        parity_mask = 0b101
        truth = [
            (parity_mask & query).bit_count() & 1
            for query in range(1 << dimension)
        ]
        truth[0] ^= 1
        for query in range(1 << dimension):
            self.assertGreaterEqual(
                two_call_self_correction_accuracy(
                    truth,
                    parity_mask=parity_mask,
                    query=query,
                ),
                Fraction(3, 4),
            )

    def test_registered_population_rederived_from_frozen_rows(self) -> None:
        path = Path("results/exp_071/raw/tensor_rows.jsonl")
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertEqual(
            registered_population_from_records(records),
            (REGISTERED_BINARY_BITS, REGISTERED_OUTPUT_ROWS),
        )

    def test_registered_hot_only_advantage_is_astronomically_small(self) -> None:
        ceiling = self_contained_advantage_ceiling()
        self.assertLess(ceiling.log2_advantage_ceiling, -8_000)
        self.assertLess(ceiling.log10_advantage_ceiling, -2_500)

    def test_practical_advantage_requires_nearly_the_binary_checkpoint(self) -> None:
        minimum = minimum_self_contained_state_bits(
            binary_coefficient_bits=REGISTERED_BINARY_BITS,
            output_rows=REGISTERED_OUTPUT_ROWS,
            advantage=Fraction(1, 100),
        )
        self.assertGreater(minimum, 5 * HOT_BITS)
        self.assertLessEqual(minimum, REGISTERED_BINARY_BITS)

    def test_audit_preserves_amplifier_and_general_probe_boundary(self) -> None:
        audit = derive_audit()
        self.assertEqual(audit["decision"], DECISION)
        self.assertFalse(audit["old_2_5_percent_assumption_used"])
        self.assertFalse(
            audit["amplifier_boundary"][
                "published_reduction_supplies_approximate_oracle"
            ]
        )
        self.assertFalse(
            audit["claim_boundary"][
                "general_nonlinear_adaptive_probe_oracle_rejected"
            ]
        )
        self.assertFalse(audit["claim_boundary"]["target_achieved"])

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            fourier_list_size_upper_bound(Fraction(0))
        with self.assertRaises(ValueError):
            self_contained_advantage_ceiling(state_bits=REGISTERED_BINARY_BITS)
        with self.assertRaises(ValueError):
            parity_agreement_count([0, 1, 0], dimension=2, advantage=Fraction(1, 4))
        with self.assertRaises(ValueError):
            registered_population_from_records([])


if __name__ == "__main__":
    unittest.main()
