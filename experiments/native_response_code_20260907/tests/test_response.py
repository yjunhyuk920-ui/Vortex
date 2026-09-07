import hashlib, sys, unittest, struct
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from response import encode,Program,factor,mobius,pulse_query,runs,run_query,HEADER
from native import bf_bits,value,scalar_bf,scalar_value,linear,scalar_linear,alphabet,inputs

def minor_rank(table,meta,L,R):
    t=np.asarray(table,dtype='<u2').reshape(L,R,-1)
    rows=[int.from_bytes(t[j].tobytes(),'little') for j in meta['basis_rows']]
    piv=meta['pivot_bits']; r=len(piv)
    M=np.array([[(v>>p)&1 for p in piv] for v in rows],np.uint8)
    k=0
    for c in range(r):
        nz=np.flatnonzero(M[k:,c])
        if not len(nz): continue
        p=k+int(nz[0]);M[[k,p]]=M[[p,k]]
        for i in range(k+1,r):
            if M[i,c]: M[i]^=M[k]
        k+=1
    return k

class TestResponse(unittest.TestCase):
    def fixture(self):
        alpha=alphabet(4); xs=inputs(alpha);rng=np.random.default_rng(33)
        table=rng.integers(0,65536,size=(256,4),dtype=np.uint16)
        blob,meta=encode(table,alpha);return alpha,xs,table,blob,meta
    def test_all_random_table_words(self):
        a,x,t,b,m=self.fixture(); p=Program.load(b)
        for i in range(len(x)):np.testing.assert_array_equal(p.query(x[i])[0],t[i])
    def test_independent_minor_certificate(self):
        a,x,t,b,m=self.fixture();self.assertEqual(minor_rank(t,m,16,16),m['rank'])
    def test_gf2_factor_exact(self):
        rows=[bytes([v]) for v in (3,5,6,0,3)]
        sel,codes,piv=factor(rows)
        for raw,c in zip(rows,codes):
            z=0
            for k,i in enumerate(sel):
                if c>>k&1:z^=int.from_bytes(rows[i],'little')
            self.assertEqual(z,int.from_bytes(raw,'little'))
        self.assertEqual(len(sel),2)
    def test_full_rank_does_not_force_long_query(self):
        rows=[(1<<i).to_bytes(2,'little') for i in range(16)]
        sel,codes,piv=factor(rows)
        self.assertEqual(len(sel),16);self.assertTrue(all(c.bit_count()==1 for c in codes))
    def test_signed_zero_and_nan_payloads_are_bits(self):
        a=alphabet(2);x=inputs(a);t=np.resize(np.array([0,0x8000,0x7fc1,0xffc2],np.uint16),(16,4))
        b,_=encode(t,a);p=Program.load(b)
        for i in range(16):np.testing.assert_array_equal(p.query(x[i])[0],t[i])
    def test_zero_rank(self):
        a=alphabet(2);x=inputs(a);t=np.zeros((16,4),np.uint16);b,meta=encode(t,a)
        p=Program.load(b);self.assertEqual(p.r,0);self.assertEqual(p.query(x[0])[1]['atoms'],0)
    def test_no_weight_or_activation_attributes(self):
        p=Program.load(self.fixture()[3]);self.assertFalse(hasattr(p,'G'));self.assertFalse(hasattr(p,'lut'))
    def test_checksum_corruption(self):
        b=bytearray(self.fixture()[3]);b[-40]^=1
        with self.assertRaises(ValueError):Program.load(bytes(b))
    def test_truncation(self):
        with self.assertRaises(ValueError):Program.load(self.fixture()[3][:-1])
    def test_bad_shapes_rehashed(self):
        b=bytearray(self.fixture()[3]);b[20:24]=struct.pack('<I',7);b[-32:]=hashlib.sha256(b[:-32]).digest()
        with self.assertRaises(ValueError):Program.load(bytes(b))
    def test_duplicate_domain_rehashed(self):
        b=bytearray(self.fixture()[3]);b[42:44]=b[40:42];b[-32:]=hashlib.sha256(b[:-32]).digest()
        with self.assertRaises(ValueError):Program.load(bytes(b))
    def test_unknown_input_is_rejected_not_rounded(self):
        a,x,t,b,m=self.fixture();p=Program.load(b);xx=x[0].copy();xx[0]=0x3fa0
        with self.assertRaises(ValueError):p.query(xx)
    def test_wrong_width(self):
        p=Program.load(self.fixture()[3])
        with self.assertRaises(ValueError):p.query([0])
    def test_mobius_involution_and_query(self):
        _,_,t,_,_=self.fixture();c=mobius(t);np.testing.assert_array_equal(mobius(c),t)
        for i in range(len(t)):np.testing.assert_array_equal(pulse_query(c,i)[0],t[i])
    def test_run_cover_exact(self):
        _,_,t,_,_=self.fixture();s,v=runs(t)
        for i in range(len(t)):np.testing.assert_array_equal(run_query(s,v,i)[0],t[i])
    def test_bf16_rounding_signed_zero(self):
        b=bf_bits(np.array([0.,-0.,1.,1.00390625,1.01171875],np.float32))
        np.testing.assert_array_equal(b,np.array([0,0x8000,0x3f80,0x3f80,0x3f82],np.uint16))
    def test_scalar_vector_rounding(self):
        rng=np.random.default_rng(7);v=rng.normal(size=1000).astype(np.float32)
        np.testing.assert_array_equal(bf_bits(v),np.array([scalar_bf(x) for x in v],np.uint16))
    def test_native_tree_scalar(self):
        rng=np.random.default_rng(91);W=bf_bits(rng.normal(size=(8,16)));X=bf_bits(rng.normal(size=(4,16)))
        result=linear(value(X),value(W))
        for i,x in enumerate(X):np.testing.assert_array_equal(result[i],scalar_linear(x,W))
    def test_domain_zero_sign_is_distinct(self):
        a=alphabet(16);self.assertEqual(len(set(map(int,a))),16);self.assertIn(0x8000,a);self.assertIn(0,a)
    def test_bounded_all_fixtures_rank_certificates(self):
        import json
        d=ROOT/'results/run'
        if not d.exists():self.skipTest('run the reference experiment first')
        for folder in d.iterdir():
            if not folder.is_dir():continue
            meta=json.loads((folder/'basis_certificate.json').read_text());p=Program.load((folder/'program.bin').read_bytes())
            table=np.load(folder/'response.npy');self.assertEqual(minor_rank(table,meta,p.L,p.R),p.r,folder.name)

if __name__=='__main__':unittest.main()
