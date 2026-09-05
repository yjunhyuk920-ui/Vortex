"""Focused controls; not a public-checkpoint or GPU acceptance suite."""
import math
import random
import sys
import unittest
from fractions import Fraction
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/native_transfer'))
from native_transfer import *

class NativeTransferTests(unittest.TestCase):
    def test_integer_ties_both_signs(self):
        for k in range(-24,25):
            x=Fraction(2*k+1,2)
            self.assertEqual(round_even(x), k if k%2==0 else k+1)

    def test_even_translation_not_odd(self):
        self.assertNotEqual(round_even(Fraction(1,2))+1,round_even(Fraction(3,2)))
        for p in range(-5,6):
            for k in range(-5,6):
                x=Fraction(p,4)
                self.assertEqual(round_even(x+2*k),round_even(x)+2*k)

    def test_oracle_signed_zero_and_subnormal(self):
        for w in (0.0,-0.0,value(1),-value(1),value(3),-value(3)):
            for x in (1.0,-1.0,0.5,-0.5):
                for a in (0.0,-0.0,value(1),-value(1)):
                    self.assertEqual(exact_fma_word(w,x,a),bits(fma(w,x,a)))

    def test_oracle_fma_not_separate_product(self):
        w=x=value(0x3f800001); a=-value(0x3f800002)
        self.assertEqual(exact_fma_word(w,x,a),bits(fma(w,x,a)))
        separate=value(bits(value(bits(w*x))+a))
        self.assertNotEqual(bits(separate),bits(fma(w,x,a)))

    def test_phase_leaf_all_small_quarters(self):
        for t in (Fraction(i,4) for i in range(-32,33)):
            s=PhaseSummary.leaf(t)
            for k in range(-32,33):
                self.assertEqual(k+s.delta[k&1],round_even(k+t))

    def test_phase_associativity_and_identity(self):
        nodes=[PhaseSummary.leaf(Fraction(i,2)) for i in range(-3,4)]
        for a in nodes:
            self.assertEqual(a.then(PhaseSummary()),a)
            for b in nodes:
                for c in nodes:
                    self.assertEqual(a.then(b).then(c),a.then(b.then(c)))

    def test_phase_native_both_signs(self):
        for sign in (-1,1):
            for word in (0x3fc00000,0x3fc00001):
                initial=sign*value(word)
                w=[2**-24,-2**-24,3*2**-24,2**-25]
                s=summarize(w,[1.0]*4,0)
                self.assertIsNotNone(s.apply(initial,0))
                self.assertEqual(bits(s.apply(initial,0)),bits(reference_dot(w,[1.0]*4,initial)))

    def test_phase_constructor_normalizes_binary32(self):
        w=[2**-24+2**-49]; x=[1.0]
        self.assertEqual(summarize(w,x,0),summarize([value(bits(w[0]))],x,0))

    def test_phase_full_prefix_guard(self):
        w=[1.,2**-23,-1.]; x=[1.]*3; initial=1.5
        s=summarize(w,x,0); k=int(initial/2**-23)
        naive=math.ldexp(k+s.delta[k&1],-23)
        self.assertNotEqual(bits(naive),bits(reference_dot(w,x,initial)))
        self.assertIsNone(s.apply(initial,0))

    def test_phase_reject_boundaries_and_bad_exponents(self):
        s=PhaseSummary()
        for a in (0.,-0.,value(1),1.,-1.,float('inf')):
            self.assertIsNone(s.apply(a,0))
        with self.assertRaises(ValueError):summarize([1.],[1.],-127)

    def test_positive_suffix_and_charged_inputs(self):
        w=[1.]*1022+[2.**80,-2.**80]; x=[1.]*1024
        index=PrefixIndex.build(w)
        y,c=suffix_dot(index,x)
        self.assertEqual(bits(y),bits(reference_dot(w,x)))
        self.assertTrue(c.certified)
        self.assertEqual(c.distinct_weight_reads,32)
        self.assertEqual(c.total_fmas,64)
        self.assertEqual(c.xmax_input_reads,1024)
        self.assertEqual(index.constructor_weight_reads,1024)
        self.assertEqual(index.metadata_bytes,82)

    def test_no_erasure_pays_all_retries(self):
        w=[1.,-1.]*512; x=[1.]*1024
        y,c=suffix_dot(PrefixIndex.build(w),x)
        self.assertEqual(bits(y),bits(reference_dot(w,x)))
        self.assertFalse(c.certified)
        self.assertEqual(c.distinct_weight_reads,1024)
        self.assertEqual(c.total_fmas,3*1024-2*32)
        self.assertEqual(c.repeated_weight_operand_reads,2*1024-32)

    def test_prefix_radius_bounds_actual_prefix(self):
        rng=random.Random(34)
        for n in (33,64,129):
            w=[value(bits(rng.randrange(-100,101)/8)) for _ in range(n)]
            x=[value(bits(rng.randrange(-100,101)/16)) for _ in range(n)]
            index=PrefixIndex.build(w)
            for r in (1,n//2,n):
                radius=index.bound(r,max(map(abs,x)))
                self.assertIsNotNone(radius)
                self.assertLessEqual(abs(reference_dot(w[:r],x[:r])),radius)

    def test_overflow_bound_falls_back_without_false_acceptance(self):
        w=[value(0x7f7fffff)]*64; x=[2.]*64
        y,c=suffix_dot(PrefixIndex.build(w),x)
        self.assertFalse(c.certified)
        self.assertEqual(bits(y),bits(reference_dot(w,x)))
        self.assertEqual(c.total_fmas,64)

    def test_suffix_non_power_two_and_empty(self):
        self.assertEqual(suffix_dot(PrefixIndex.build([]),[])[0],0.)
        for n in (1,31,33,97):
            w=[1.,-1.]*(n//2)+([1.] if n%2 else [])
            y,c=suffix_dot(PrefixIndex.build(w),[1.]*n)
            self.assertEqual(bits(y),bits(reference_dot(w,[1.]*n)))
            self.assertLess(c.total_fmas,5*n)

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):PrefixIndex.build([1.],0)
        with self.assertRaises(ValueError):PrefixIndex.build([float('nan')])
        with self.assertRaises(ValueError):suffix_dot(PrefixIndex.build([1.]),[])
        with self.assertRaises(ValueError):suffix_dot(PrefixIndex.build([1.]),[float('inf')])
        with self.assertRaises(ValueError):summarize([1.],[],0)
        with self.assertRaises(ValueError):preimage(1.,1.,MAX_FINITE_RANK,MIN_FINITE_RANK)
        with self.assertRaises(ValueError):suffix_preimage([],[],0x7f800000)

    def test_total_order_including_signed_zero(self):
        words=[0xff7fffff,0xbf800000,0x80000001,0x80000000,0,1,0x3f800000,0x7f7fffff]
        self.assertEqual(sorted(map(rank_word,words)),list(map(rank_word,words)))
        for w in words:self.assertEqual(unrank_word(rank_word(w)),w)

    def test_direct_preimage_entire_interval_matches_native_search(self):
        for w,x in ((1.,1.),(-1.,1.),(0.,1.),(-0.,1.),(value(1),.5),(2.**80,1.)):
            for ow in (0,0x80000000,1,0x80000001,0x3f800000,0x7f7fffff,0xff7fffff):
                r=rank_word(ow)
                a=preimage(w,x,r,r); b=preimage_direct(w,x,r,r)
                self.assertEqual(None if a is None else a[:2],None if b is None else b[:2])
                if b is not None:self.assertLessEqual(b[3],64)

    def test_direct_midpoint_tie_parity(self):
        for word in (0x3f800000,0x3f800001,0xbf800000,0xbf800001):
            r=rank_word(word)
            slow=preimage(2**-24,1.,r,r)
            fast=preimage_direct(2**-24,1.,r,r)
            self.assertEqual(None if slow is None else slow[:2],None if fast is None else fast[:2])
            if fast is not None:
                self.assertEqual(fast[2],'exact_rounding_cell')
                self.assertEqual(fast[3],0)

    def test_inverse_suffix_accepts_only_its_own_preimage(self):
        w=[2.,-4.,3.];x=[1.,.5,2.];initial=.25
        word=bits(reference_dot(w,x,initial)); inv=suffix_preimage(w,x,word)
        self.assertIsNotNone(inv)
        self.assertLessEqual(inv[0],rank_word(bits(initial)))
        self.assertLessEqual(rank_word(bits(initial)),inv[1])
        for r in (inv[0],inv[1],(inv[0]+inv[1])//2):
            self.assertEqual(bits(reference_dot(w,x,value(unrank_word(r)))),word)
        for r in (inv[0]-1,inv[1]+1):
            if MIN_FINITE_RANK<=r<=MAX_FINITE_RANK:
                self.assertNotEqual(bits(reference_dot(w,x,value(unrank_word(r)))),word)

if __name__=='__main__':unittest.main()
