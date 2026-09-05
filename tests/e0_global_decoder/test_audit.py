import importlib.util
from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("audit", ROOT / "experiments/e0_global_decoder/audit.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class AuditTests(unittest.TestCase):
    def test_rank_against_row_span_enumeration(self):
        for m, n in ((2, 2), (2, 3)):
            for w in range(1 << (m*n)):
                rows = [(w >> (i*n)) & ((1 << n)-1) for i in range(m)]
                span = {0}
                for row in rows:
                    span |= {x ^ row for x in list(span)}
                self.assertEqual(1 << a.gf2_rank(w, m, n), len(span))

    def test_query_masks_against_explicit_rectangles(self):
        for m, n in ((2, 2), (2, 3)):
            self.assertEqual(len(set(a.queries(m, n))), ((1 << m)-1)*((1 << n)-1))
            for u in range(1 << m):
                for v in range(1 << n):
                    got = a.outer_mask(u, v, m, n)
                    for i in range(m):
                        for j in range(n):
                            self.assertEqual((got >> (i*n+j)) & 1, ((u >> i) & 1)*((v >> j) & 1))

    def test_kernel_obstruction_exact(self):
        out = a.kernel_obstruction()
        self.assertEqual(out["possible_complement_products"], [-3, 1, 5])
        self.assertEqual(out["unit_magnitude_compatible_products"], [1])

    def test_all_kernel_integrity_controls(self):
        for shape in ((2, 2), (2, 3)):
            out = a.verify_kernel(*shape)
            self.assertEqual(out["ordered_pair_kernel_checks"], out["matrices"]**2)

    def test_proof_must_not_reject_redundancy_words_or_restricted_domain(self):
        self.assertEqual(a.proof_scope(2, 2, 4, 1, 0, True, 2), "IMPOSSIBLE_IN_REGISTERED_SCOPE")
        self.assertEqual(a.proof_scope(2,2,4,1,0,True,2,False), 'NO_CONCLUSION_FROM_THIS_THEOREM')
        for case in [(2,2,5,1,0,True,2), (2,3,7,1,0,True,2), (25,108,50,64,0,True,2),
                     (2,2,4,1,1,True,2), (2,2,4,1,0,False,2), (2,2,4,1,0,True,3)]:
            self.assertEqual(a.proof_scope(*case), "NO_CONCLUSION_FROM_THIS_THEOREM")

    def test_five_bit_positive_and_nontrivial_state(self):
        out = a.controls()
        self.assertEqual(out["exact_query_cases"], 144)
        self.assertEqual(out["dynamic_gf2_state_transitions"], 8192)
        self.assertEqual(out["state_mismatches"], 0)

    def test_injective_five_bit_code(self):
        self.assertEqual(len({a.encode_five_bits(w) for w in range(16)}), 16)

    def test_faults_detected_for_every_cell(self):
        self.assertEqual(len(a.controls()["fault_injection_witnesses"]), 5)

    def test_native_partial_does_not_equal_parity(self):
        out = a.native_boundary()
        self.assertEqual(out["small_binary_ordered_float32_cases"], 1488)
        self.assertEqual(out["parity_does_not_determine_native_sum"]["different_sums"], [0,2])

    def test_native_order_cannot_be_reassociated(self):
        c = a.native_boundary()["reassociation_counterexample"]
        self.assertNotEqual(c["left_fp32"], c["right_fp32"])

    def test_capacity_boundary_is_diagnostic(self):
        b = a.budget()
        self.assertEqual(b["nominal_fraction"], "8/675")
        self.assertEqual(b["nominal_budget_slack_bytes"], 0)
        self.assertEqual(b["conditional_one_additional_byte_over_budget_factor"], "17/16")

    def test_no_hardware_or_checkpoint_promotion(self):
        out = a.build_result()["overall"]
        for key in ("405b_execution", "physical_8gib", "same_machine_4b_latency"):
            self.assertEqual(out[key], "NOT_TESTED")
        self.assertEqual(out["general_impossibility"], "NOT_ESTABLISHED")

    def test_invalid_input_fails_closed(self):
        for w in (-1, 16):
            with self.assertRaises(ValueError): a.encode_five_bits(w)
        with self.assertRaises(ValueError): a.decode_five_bits((0,0,0,0,0), 7)
        with self.assertRaises(ValueError): a.outer_mask(4, 0, 2, 2)
        with self.assertRaises(ValueError): a.verify_kernel(4, 4)

    def test_result_regeneration(self):
        path = ROOT / "results/e0_global_decoder/summary.json"
        if path.exists():
            self.assertEqual(json.loads(path.read_text()), a.build_result())



# Deliberately independent of the symbolic backend: check the two-probe formula
# on every 7-cell word and every choice of roots, children, and legal leaves.
class SynthesisGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('synthesis', ROOT / 'experiments/e0_global_decoder/synthesis.py')
        cls.s = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.s)

    def test_packed_formula_against_branch_execution(self):
        words = 128
        all_bits = (1 << words)-1
        cells = [sum(((w>>i)&1)<<w for w in range(words)) for i in range(7)]
        cases = 0
        for root in range(7):
            for zero_child in range(7):
                for one_child in range(7):
                    x,y,z = cells[root],cells[zero_child],cells[one_child]
                    for leaves in range(8):
                        u,v,w = (leaves>>0)&1,(leaves>>1)&1,(leaves>>2)&1
                        packed = ((~x & y & (all_bits*u)) | (x & ((~z & (all_bits*v)) | (z & (all_bits*w))))) & all_bits
                        for code in range(words):
                            bit = (code>>root)&1
                            child = one_child if bit else zero_child
                            second = (code>>child)&1
                            expected = ((w if second else v) if bit else (u if second else 0))
                            self.assertEqual((packed>>code)&1, expected)
                            cases += 1
        self.assertEqual(cases,351232)

    def test_exact_formula_hash_matches_original_run(self):
        import hashlib
        record = json.loads((ROOT/'results/e0_global_decoder/solver_seven_bit.json').read_text())
        self.assertEqual(hashlib.sha256(self.s.build_smt().encode()).hexdigest(),record['formula_sha256'])

    def test_solver_unknown_is_not_unsat(self):
        record = json.loads((ROOT/'results/e0_global_decoder/solver_seven_bit.json').read_text())
        self.assertEqual(record['result'],'unknown')
        self.assertEqual(record['classification'],'INCONCLUSIVE')

    def test_positive_solver_control_was_satisfiable(self):
        record = json.loads((ROOT/'results/e0_global_decoder/solver_positive.json').read_text())
        self.assertEqual(record['result'],'sat')

if __name__ == "__main__": unittest.main()
