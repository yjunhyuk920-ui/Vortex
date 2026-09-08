from __future__ import annotations

import random
import unittest

from vortex_runtime.implicit_program_carrier_gate import (
    alias_router_square_audit,
    compile_alias_router,
    compile_patricia_program,
    compile_rank_normal_gauge,
    direct_gf2_matvec_mask,
    gf2_inverse,
    gf2_matvec,
    literal_transformed_hadamard,
    patricia_square_audit,
    query_alias_router,
    query_patricia_program,
    query_rank_normal_gauge,
    rank_normal_hadamard_cost_audit,
    registered_alias_router_audit,
    verify_rank_normal_gauge,
)


class ImplicitProgramCarrierGateTests(unittest.TestCase):
    def test_alias_router_is_exact_for_every_small_query(self):
        rng = random.Random(26090911)
        matrix = [[rng.randrange(2) for _ in range(9)] for _ in range(7)]
        router = compile_alias_router(matrix, (4, 5))
        for vector in range(1 << 9):
            got, _ = query_alias_router(router, vector)
            self.assertEqual(got, direct_gf2_matvec_mask(matrix, vector))

    def test_alias_logical_addresses_are_fixed_row_block_pairs(self):
        matrix = [[(i * 3 + j) & 1 for j in range(8)] for i in range(5)]
        router = compile_alias_router(matrix, (4, 4))
        self.assertEqual(router.rows * len(router.block_widths), 10)
        self.assertEqual(len(router.aliases), 5)
        self.assertTrue(all(len(row) == 2 for row in router.aliases))

    def test_alias_square_has_favorable_arithmetic_but_source_scale_information(self):
        audit = alias_router_square_audit()
        self.assertLess(audit["candidate_operation_fraction"], 0.1)
        self.assertTrue(audit["removes_at_least_90_percent_favorable_arithmetic"])
        self.assertEqual(audit["alias_information_lower_over_source"], 1.0)
        self.assertGreater(audit["descriptor_over_binary_source_if_64_per_alias"], 6.0)

    def test_registered_alias_descriptors_do_not_fit_eight_gib(self):
        audit = registered_alias_router_audit()
        self.assertEqual(audit["matrix_count"], 883)
        self.assertFalse(audit["fits_8gib_if_all_descriptors_hot"])
        self.assertGreater(audit["descriptor_gib_if_64_per_alias"], 100.0)

    def test_patricia_program_is_exact_for_every_small_query(self):
        rng = random.Random(26090912)
        matrix = [[rng.randrange(2) for _ in range(11)] for _ in range(8)]
        program = compile_patricia_program(matrix, block_width=6)
        for vector in range(1 << 11):
            got, _ = query_patricia_program(program, vector)
            self.assertEqual(got, direct_gf2_matvec_mask(matrix, vector))

    def test_patricia_handles_duplicate_row_patterns(self):
        matrix = [
            [1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
        ]
        program = compile_patricia_program(matrix, block_width=6)
        for vector in range(1 << 6):
            got, stats = query_patricia_program(program, vector)
            self.assertEqual(got, direct_gf2_matvec_mask(matrix, vector))
        self.assertEqual(stats["leaf_memberships"], 4)

    def test_patricia_favorable_arithmetic_but_edge_label_traffic_is_source_scale(self):
        audit = patricia_square_audit()
        self.assertLess(audit["favorable_event_fraction"], 0.1)
        self.assertTrue(audit["removes_at_least_90_percent_favorable_arithmetic"])
        self.assertGreater(audit["edge_label_over_binary_source"], 1.9)
        self.assertFalse(audit["topology_and_membership_bits_included"])

    def test_rank_normal_compiler_reaches_exact_normal_form_on_rectangles(self):
        rng = random.Random(26090913)
        for rows, columns in ((3, 5), (5, 3), (5, 5), (6, 7)):
            for _ in range(10):
                matrix = [[rng.randrange(2) for _ in range(columns)] for _ in range(rows)]
                gauge = compile_rank_normal_gauge(matrix)
                self.assertTrue(verify_rank_normal_gauge(matrix, gauge))
                for i in range(rows):
                    for j in range(columns):
                        self.assertEqual(gauge.normal[i][j], int(i == j and i < gauge.rank))

    def test_rank_normal_encoded_query_matches_direct_encoded_output(self):
        rng = random.Random(26090914)
        matrix = [[rng.randrange(2) for _ in range(6)] for _ in range(5)]
        gauge = compile_rank_normal_gauge(matrix)
        for _ in range(64):
            vector = [rng.randrange(2) for _ in range(6)]
            got, expected = query_rank_normal_gauge(matrix, gauge, vector)
            self.assertEqual(got, expected)

    def test_literal_transformed_hadamard_restores_dense_maps_exactly(self):
        w1 = [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ]
        w2 = [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ]
        inv1 = gf2_inverse(w1)
        inv2 = gf2_inverse(w2)
        for z1_mask in range(1 << 4):
            z1 = [(z1_mask >> j) & 1 for j in range(4)]
            a = gf2_matvec(w1, z1)
            self.assertEqual(gf2_matvec(inv1, a), z1)
            for z2_mask in range(1 << 4):
                z2 = [(z2_mask >> j) & 1 for j in range(4)]
                b = gf2_matvec(w2, z2)
                self.assertEqual(gf2_matvec(inv2, b), z2)
                expected = [x & y for x, y in zip(a, b, strict=True)]
                self.assertEqual(literal_transformed_hadamard(w1, w2, z1, z2), expected)

    def test_rank_normal_isolated_linear_win_is_lost_in_literal_hadamard_micrograph(self):
        audit = rank_normal_hadamard_cost_audit()
        self.assertTrue(audit["isolated_removes_at_least_90_percent"])
        self.assertFalse(audit["whole_micrograph_removes_at_least_90_percent"])
        self.assertGreater(audit["whole_micrograph_candidate_fraction"], 1.0)
        self.assertEqual(audit["dense_maps_restored_inside_transformed_nonlinearity"], 2)


if __name__ == "__main__":
    unittest.main()
