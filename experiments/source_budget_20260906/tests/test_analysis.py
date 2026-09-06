import itertools
import sys
from fractions import Fraction
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from analyze import balanced_table_size, fixed_block_cost, round_dyadic, bf16_value, native_witnesses, interval_gate

class Checks(unittest.TestCase):
    def test_explicit_format_count(self):
        self.assertEqual(fixed_block_cost(64,4,2)['table_bytes'],16*16*4)
    def test_partition_exact_minimum(self):
        # Every positive ordered partition for n<=10; no sampled instances.
        for n in range(1,11):
            for q in (2,3,5):
                minima={}
                for mask in range(1<<(n-1)):
                    lengths=[]; start=0
                    for j in range(1,n):
                        if mask>>(j-1)&1:
                            lengths.append(j-start); start=j
                    lengths.append(n-start)
                    k=len(lengths); cost=4*sum(q**b for b in lengths)
                    minima[k]=min(minima.get(k,cost),cost)
                for k,cost in minima.items():
                    self.assertEqual(balanced_table_size(n,k,q),cost)
    def test_uniform_hot_cache_formula(self):
        for q,b,blocks in ((2,2,3),(3,1,2)):
            entries=q**b
            for allocation in itertools.product(range(entries+1),repeat=blocks):
                hot=4*sum(allocation)
                expected=sum(Fraction(4*(entries-h),entries) for h in allocation)
                self.assertEqual(Fraction(fixed_block_cost(b*blocks,b,q,hot)['expected_cold_lower_bound']),expected)
    def test_invalid_inputs(self):
        for args in ((0,1,2),(3,0,2),(3,4,2),(3,1,1)):
            with self.assertRaises(ValueError): balanced_table_size(*args)
        with self.assertRaises(ValueError): fixed_block_cost(5,2,2)
        with self.assertRaises(ValueError): interval_gate(2**24)
    def test_target_count(self):
        c=fixed_block_cost(405000000000,32,2,8*2**30)
        self.assertEqual(c['table_bytes'],217432719360000000000)
        self.assertEqual(c['query_fraction'],'1/16')
        self.assertEqual(c['expected_cold_lower_bound'],'50624999998')
    def test_tenfold_optimistic_block20(self):
        c=fixed_block_cost(405000000000,20,2)
        self.assertEqual(c['query_fraction'],'1/10')
        self.assertEqual(c['table_to_weight_ratio'],'524288/5')
    def test_independent_rounding(self):
        for i in range(-20000,20001):
            self.assertEqual(Fraction(bf16_value(float(i))),round_dyadic(Fraction(i),8))
    def test_ties_even(self):
        self.assertEqual(round_dyadic(Fraction(257),8),256)
        self.assertEqual(round_dyadic(Fraction(259),8),260)
        self.assertEqual(round_dyadic(Fraction(-257),8),-256)
    def test_native_subtree_witness(self):
        a,b=native_witnesses()['rows']
        self.assertEqual(a['bf16_left'],b['bf16_left'])
        self.assertNotEqual(a['reference_final'],b['reference_final'])
        self.assertNotEqual(a['reference_final'],a['premature_final'])
    def test_interval_theorem_endpoints(self):
        for n in range(1,65):
            bound=interval_gate(n)['derived_terms_required_at_least']
            for k in range(bound):
                R=n-k
                for s in range(-k,k+1,2):
                    self.assertNotEqual(round_dyadic(Fraction(s-R),8),round_dyadic(Fraction(s+R),8))
    def test_interval_16384(self):
        c=interval_gate(16384)
        self.assertEqual(c['derived_terms_required_at_least'],16321)
        self.assertEqual(c['all_positive_first_endpoint_certificate_after_terms'],16368)
    def test_other_precision_endpoint_bounds(self):
        for p in range(3,9):
            for n in range(1,25):
                bound=interval_gate(n,p)['derived_terms_required_at_least']
                for k in range(bound):
                    R=n-k
                    for s in range(-k,k+1,2):
                        self.assertNotEqual(round_dyadic(Fraction(s-R),p),round_dyadic(Fraction(s+R),p))

if __name__=='__main__': unittest.main()
