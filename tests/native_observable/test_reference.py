from fractions import Fraction as F
from pathlib import Path
import sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/native_observable'))
from reference import BFloat,PrefixSource,generate,oracle,p2,rounded,stored,interval_cell,prefix_interval

class ReferenceTests(unittest.TestCase):
    def test_even_tie(self):
        self.assertEqual(stored(F(1)+p2(-8)),1)
    def test_odd_tie(self):
        self.assertEqual(stored(F(1)+3*p2(-8)),1+p2(-6))
    def test_negative_tie(self):
        self.assertEqual(stored(-F(1)-p2(-8)),-1)
    def test_cell_straddles_tie(self):
        self.assertIsNone(interval_cell(1+p2(-8),1+p2(-8)+p2(-24)))
    def test_different_internal_same_store(self):
        self.assertEqual(interval_cell(F(1),1+p2(-23)),1)
    def test_reversed_interval(self):
        with self.assertRaises(ValueError): interval_cell(F(2),F(1))
    def test_source_reveal_cap(self):
        s=PrefixSource([BFloat(1,0,127)]*8)
        self.assertEqual([s.next_bit(0) for _ in range(7)],[1]*7)
        with self.assertRaises(ValueError): s.next_bit(0)
    def test_source_counted_headers(self):
        s=PrefixSource([BFloat(1,0,0)]*8)
        r=generate(s,[BFloat(1,0,0)]*8)
        self.assertEqual(r['value'],8)
        self.assertEqual(r['header_bits'],72)
        self.assertLessEqual(r['mantissa_bits'],56)
        self.assertEqual(r['preparation_bits'],128)
    def test_prefix_signed(self):
        lo,hi=prefix_interval((-1,0),0,0)
        self.assertEqual((lo,hi),(-F(255,128),-F(1)))
    def test_no_inserted_rounding_contract(self):
        # A private FP32 value cannot be collapsed if a later operation still uses it.
        a,b=F(1),1+p2(-23)
        self.assertEqual(stored(a),stored(b))
        self.assertNotEqual(stored(a-1),stored(b-1))
    def test_guarded_domain(self):
        with self.assertRaises(ValueError):
            generate(PrefixSource([BFloat(1,31,0)]*8),[BFloat(1,0,0)]*8)
    def test_mixed_sign(self):
        w=[BFloat(1 if i%2 else -1,i-4,13*i) for i in range(8)]
        x=[BFloat(1,3-i,7*i) for i in range(8)]
        self.assertEqual(generate(PrefixSource(w),x)['value'],oracle(w,x))

if __name__=='__main__': unittest.main()
