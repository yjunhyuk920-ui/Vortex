import itertools
import unittest

from producer import (
    bits, classify_gauges, compile_transport, decode_rows, dense_reference,
    execute_transport, matvec, operation_ledger,
)


class GaugeAuditTests(unittest.TestCase):
    def test_separate_diagonal_gauges_are_allowed(self):
        a = [[0, 2], [1, 0]]
        b = [[0, 1], [2, 0]]
        c = [[0, 2], [2, 0]]
        result = classify_gauges(a, b, c, 3)
        self.assertEqual(result["status"], "CERTIFIED")
        self.assertEqual(result["permutation"], [1, 0])
        for x, y in itertools.product(itertools.product(range(3), repeat=2), repeat=2):
            left = matvec(c, [u*v % 3 for u,v in zip(x,y)], 3)
            right = [u*v % 3 for u,v in zip(matvec(a,x,3), matvec(b,y,3))]
            self.assertEqual(left, right)

    def test_dense_gauge_returns_real_basis_witness(self):
        a = [[1,1],[0,1]]
        b = [[1,0],[0,1]]
        c = [[1,0],[0,1]]
        result = classify_gauges(a,b,c,2)
        self.assertEqual(result["status"], "COUNTEREXAMPLE")
        x,y = result["x"],result["y"]
        left = matvec(c,[u*v for u,v in zip(x,y)],2)
        right = [u*v for u,v in zip(matvec(a,x,2),matvec(b,y,2))]
        self.assertNotEqual(left,right)

    def test_column_permutation_is_not_native_reassociation(self):
        weights = [[2**24,1,-2**24,0]]
        x = [1,1,1,1]
        plan = compile_transport(weights,[0,2,1,3],[0])
        encoded = [x[j] for j in plan.column_order]
        self.assertEqual(bits(dense_reference(weights,x)[0]),0)
        self.assertEqual(bits(dense_reference(plan.weights,encoded)[0]),0x3f800000)
        self.assertEqual(bits(execute_transport(plan,encoded)[0]),0)

    def test_original_tree_slots_survive_nonpower2_and_row_order(self):
        weights = [[2**24,1,-2**24],[1,2,3]]
        x = [1,1,1]
        for columns in itertools.permutations(range(3)):
            plan = compile_transport(weights,columns,[1,0])
            result = decode_rows(plan,execute_transport(plan,[x[j] for j in columns]))
            self.assertEqual(list(map(bits,result)),list(map(bits,dense_reference(weights,x))))

    def test_signed_zero_storage_is_transportable_in_declared_abi(self):
        weights = [[-0.0,-0.0],[0.0,-0.0]]
        x = [1.0,1.0]
        plan = compile_transport(weights,[1,0],[1,0])
        result = decode_rows(plan,execute_transport(plan,[1.0,1.0]))
        self.assertEqual(list(map(bits,result)),list(map(bits,dense_reference(weights,x))))

    def test_bad_permutation_is_not_admitted(self):
        with self.assertRaises(ValueError):
            compile_transport([[1,2]],[0,0],[0])

    def test_cost_does_not_claim_native_arithmetic_removal(self):
        cost = operation_ledger(16384,16384)
        self.assertEqual(cost["runtime_multiplications"],268435456)
        self.assertEqual(cost["runtime_padded_additions"],268419072)
        self.assertEqual(cost["native_arithmetic_fraction_numerator"],1)
        self.assertEqual(cost["native_arithmetic_fraction_denominator"],1)
        self.assertFalse(cost["target_hardware_measured"])


if __name__ == "__main__":
    unittest.main()
