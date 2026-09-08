import sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from envelope import Plan,execute,direct,encode,decode
import oracle

class ExactEnvelopeTests(unittest.TestCase):
    def test_all_finite_bf16_roundtrip(self):
        a=np.arange(65536,dtype=np.uint16);a=a[(a&0x7f80)!=0x7f80]
        np.testing.assert_array_equal(encode(decode(a)),a)
    def test_random_nonpower_two(self):
        r=np.random.default_rng(31)
        for m,n in [(1,1),(3,5),(16,17),(33,9)]:
            w=encode(r.normal(size=(m,n)));p=Plan.build(w,8)
            for _ in range(8):
                x=encode(r.normal(size=n));y,s=execute(p,x,lambda a,b:w[a:b])
                np.testing.assert_array_equal(y,direct(w,x))
    def test_broadcast_without_weight_read(self):
        r=np.random.default_rng(7);w=encode(r.normal(size=(256,64))*2**-22)
        w[:,0]=encode(np.ones(256));x=encode(np.ones(64))
        def forbidden(*_):raise AssertionError('cold read in positive control')
        y,s=execute(Plan.build(w),x,forbidden)
        self.assertEqual(s['accepted_rows'],256)
        np.testing.assert_array_equal(y,direct(w,x))
    def test_source_roundtrip(self):
        w=encode(np.arange(30).reshape(6,5)/8)
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'p.npz';Plan.build(w,4).save(path);p=Plan.load(path)
            x=encode(np.ones(5));y,s=execute(p,x,lambda a,b:w[a:b])
            np.testing.assert_array_equal(y,direct(w,x))
    def test_negative_input_monotonicity(self):
        w=encode(np.array([[1,-3],[2,-2],[-5,7.5]]));x=encode(np.array([-8,-2]))
        y,s=execute(Plan.build(w),x,lambda a,b:w[a:b])
        np.testing.assert_array_equal(y,direct(w,x))
    def test_fraction_oracle(self):
        r=np.random.default_rng(912)
        w=encode(r.normal(size=(12,13)));x=encode(r.normal(size=13))
        np.testing.assert_array_equal(direct(w,x),np.array([oracle.dot(row,x) for row in w],dtype=np.uint16))
    def test_zero_never_unsafe_broadcast(self):
        w=np.array([[0,0],[0x8000,0x8000]],dtype=np.uint16);x=encode(np.ones(2))
        y,s=execute(Plan.build(w),x,lambda a,b:w[a:b]);self.assertEqual(s['accepted_groups'],0)
        self.assertEqual(list(y),[0,0x8000])
    def test_nonfinite_intermediate(self):
        w=np.array([[0x7f7f,0xff7f],[0x7f7f,0x7f7f]],dtype=np.uint16)
        x=np.array([0x7f7f,0x7f7f],dtype=np.uint16)
        y,s=execute(Plan.build(w),x,lambda a,b:w[a:b]);self.assertEqual(s['accepted_groups'],0)
        np.testing.assert_array_equal(y,direct(w,x))
    def test_invalid_input_rejected(self):
        w=encode(np.ones((3,2)));p=Plan.build(w)
        with self.assertRaises(ValueError):execute(p,np.array([0x7fc0,0],dtype=np.uint16),lambda a,b:w[a:b])
        with self.assertRaises(ValueError):Plan.build(np.array([[0x7f80]],dtype=np.uint16))
    def test_distinct_output_packet_must_not_certify(self):
        w=encode(np.arange(256,dtype=np.float32)[:,None]);x=encode(np.ones(1))
        y,s=execute(Plan.build(w),x,lambda a,b:w[a:b]);self.assertEqual(s['accepted_groups'],0)
    def test_partial_final_packet(self):
        w=encode(np.ones((257,5)));x=encode(np.ones(5))
        y,s=execute(Plan.build(w),x,lambda a,b:w[a:b]);self.assertEqual(s['accepted_groups'],2)
        self.assertEqual(s['accepted_rows'],257)
    def test_reader_shape_rejected(self):
        w=encode(np.arange(10).reshape(2,5));x=encode(np.ones(5))
        with self.assertRaises(ValueError):execute(Plan.build(w),x,lambda a,b:w[:1])

if __name__=='__main__':unittest.main()
