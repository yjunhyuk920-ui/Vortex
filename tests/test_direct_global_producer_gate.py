from __future__ import annotations

import random
import unittest

from vortex_runtime.direct_global_producer_gate import (
    MatrixBlock,
    compile_gf2_elimination,
    derive_direct_global_producer_audit,
    direct_gf2_reference,
    elimination_program_cost,
    execute_gf2_elimination,
    global_vector_route_gate,
    lower_triangular_elimination_cost,
    reconstructive_code_screen,
    registered_non_embedding_blocks,
    rounding_encoded_row,
    rounding_gadget_audit,
    vector_route_capacity_dimension,
    vector_route_minimum_span_per_query,
)


class DirectGlobalProducerGateTests(unittest.TestCase):
    def test_p1_reconstructive_code_fails_90_percent_gate(self) -> None:
        for bits in (1, 4, 16):
            result = reconstructive_code_screen(bits)
            self.assertTrue(result["rejected_as_core"])
            self.assertFalse(result["passes_90_percent_read_removal"])
            self.assertFalse(result["passes_90_percent_arithmetic_removal"])

    def test_rounding_gadget_has_zero_exact_sum_and_bit_output(self) -> None:
        for bit in (0, 1):
            result = rounding_gadget_audit(bit)
            self.assertEqual(result["exact_integer_sum"], 0)
            self.assertTrue(result["matches"])

    def test_rounding_gadget_embeds_arbitrary_small_matvec_rows(self) -> None:
        for width in range(1, 6):
            for row in range(1 << width):
                row_bits = [(row >> bit) & 1 for bit in range(width)]
                for query in range(1 << width):
                    query_bits = [(query >> bit) & 1 for bit in range(width)]
                    result = rounding_encoded_row(row_bits, query_bits)
                    self.assertEqual(result["exact_integer_sum"], 0)
                    self.assertTrue(result["matches"])
                    self.assertEqual(
                        result["decoded_parity_from_rounded_result"],
                        (row & query).bit_count() & 1,
                    )

    def test_registered_global_vector_route_single_tuple_span(self) -> None:
        blocks = registered_non_embedding_blocks()
        self.assertEqual(len(blocks), 883)
        span = vector_route_minimum_span_per_query(blocks)
        self.assertEqual(span, 19_997_952)
        gate = global_vector_route_gate()
        self.assertEqual(gate["single_tuple_minimum_probes"], 312_468)

    def test_route_capacity_requires_one_right_dimension_per_block(self) -> None:
        blocks = registered_non_embedding_blocks()
        mandatory = vector_route_minimum_span_per_query(blocks)
        self.assertFalse(
            vector_route_capacity_dimension(blocks, mandatory - 1)["feasible_nonempty_route"]
        )
        at_min = vector_route_capacity_dimension(blocks, mandatory)
        self.assertTrue(at_min["feasible_nonempty_route"])
        self.assertEqual(at_min["sum_right_dimensions_upper"], 883)

    def test_route_capacity_greedy_matches_tiny_exhaustive_allocation(self) -> None:
        blocks = (
            MatrixBlock("a", rows=2, columns=4),
            MatrixBlock("b", rows=3, columns=3),
            MatrixBlock("c", rows=5, columns=2),
        )
        mandatory = sum(block.rows for block in blocks)
        for budget in range(mandatory, mandatory + 20):
            actual = vector_route_capacity_dimension(blocks, budget)
            brute = 0
            for ka in range(1, 5):
                for kb in range(1, 4):
                    for kc in range(1, 3):
                        if 2 * ka + 3 * kb + 5 * kc <= budget:
                            brute = max(brute, ka + kb + kc)
            self.assertEqual(actual["sum_right_dimensions_upper"], brute)

    def test_global_route_gate_does_not_overclaim_target_rejection(self) -> None:
        gate = global_vector_route_gate()
        self.assertTrue(gate["previous_probe_count_is_proof_rejected"])
        self.assertTrue(gate["minimum_probe_count_is_only_count_feasible"])
        self.assertFalse(gate["rejects_registered_target"])
        self.assertFalse(gate["causal_reachability_of_complete_tuple_family_proved"])
        self.assertFalse(gate["native_finite_word_lift_proved"])
        self.assertIn("8 GiB advice", gate["canonical_cell_pool_scope"])

    def test_arbitrary_gf2_elimination_producer_is_exact(self) -> None:
        rng = random.Random(260909)
        for rows, columns in ((3, 4), (4, 4), (5, 3), (6, 7)):
            for _ in range(20):
                matrix = tuple(rng.randrange(1 << columns) for _ in range(rows))
                program = compile_gf2_elimination(matrix, columns)
                for query in range(1 << min(columns, 5)):
                    self.assertEqual(
                        execute_gf2_elimination(program, query),
                        direct_gf2_reference(matrix, query),
                    )

    def test_elimination_constructive_control_fails_90_percent_on_frozen_family(self) -> None:
        for width in (8, 16, 32, 64):
            matrix = tuple((1 << (row + 1)) - 1 for row in range(width))
            program = compile_gf2_elimination(matrix, width)
            cost = elimination_program_cost(program)
            self.assertFalse(cost["removes_at_least_90_percent_operations"])

        target = lower_triangular_elimination_cost(16_384)
        self.assertEqual(target["exact_operation_fraction"], "16385/32768")
        self.assertFalse(target["removes_at_least_90_percent_operations"])

    def test_audit_keeps_mission_open(self) -> None:
        audit = derive_direct_global_producer_audit()
        self.assertFalse(audit["claim_boundary"]["arbitrary_checkpoint_native_producer_constructed"])
        self.assertTrue(audit["claim_boundary"]["arbitrary_binary_matrix_elimination_producer_constructed"])
        self.assertFalse(audit["claim_boundary"]["target_405b_executed"])
        self.assertEqual(audit["obligations"]["O1"], "OPEN")
        self.assertEqual(audit["obligations"]["O5"], "OPEN")


if __name__ == "__main__":
    unittest.main()
