import unittest,sys,struct
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adjoint import Builder,Program,native_rounder,transpose_binary_matrix,HEAD,NODE
from run import fraction_round,planes,unpack,rational_rank,scope_run

class TestAdjoint(unittest.TestCase):
    def test_single_selector(self):
        b=Builder(1,1);p=b.wrap([b.input(0)])
        self.assertEqual(p.extract([0]),[0]);self.assertEqual(p.extract([1]),[1])
    def test_shared_outputs_and_cancellation(self):
        b=Builder(2,3);a,c=b.input(0),b.input(1);p=b.wrap([a,b.and_(a,c),b.xor(c,c)])
        self.assertEqual(p.extract([1,1]),[1,1,0])
    def test_zero_scalar_is_not_zero_extraction(self):
        p=transpose_binary_matrix([[1,1],[1,0]])
        self.assertEqual(p.scalar([1,0],[0,0]),0)
        self.assertEqual(p.extract([1,0]),[1,1])
    def test_probe_identity(self):
        p=transpose_binary_matrix([[1,0,1],[0,1,1]])
        for x in range(8):
            v=[(x>>i)&1 for i in range(3)]
            self.assertEqual(p.extract(v),[p.scalar(v,[1,0]),p.scalar(v,[0,1])])
    def test_roundtrip(self):
        p,_=native_rounder();raw=p.to_bytes();self.assertEqual(Program.from_bytes(raw).to_bytes(),raw)
    def test_bad_format(self):
        for raw in (b'',b'SLA0'+b'\0'*20):
            with self.assertRaises(ValueError):Program.from_bytes(raw)
    def test_reject_selector_product_opcode(self):
        p=transpose_binary_matrix([[1]]);p.linear.append((4,1,1,0));p.root=len(p.linear)-1
        with self.assertRaises(ValueError):p.validate()
    def test_reject_cycles(self):
        p=transpose_binary_matrix([[1]]);p.coeff[0]=(2,1,1,0)
        with self.assertRaises(ValueError):p.validate()
    def test_reject_out_of_domain_bit_values(self):
        p=transpose_binary_matrix([[1]])
        with self.assertRaises(ValueError):p.extract([2])
        with self.assertRaises(ValueError):transpose_binary_matrix([[3]])
    def test_rounding_ties_and_signed_zeros(self):
        p,_=native_rounder()
        words=np.array([0,0x80000000,0x3f808000,0x3f818000,0x7f7fffff,0x00008000,0x80008000,0x7fc00001],dtype=np.uint32)
        out=p.extract(planes(words,32),len(words));actual=unpack(out[:16],len(words),16)
        self.assertEqual(actual.tolist(),[fraction_round(int(w)) for w in words])
        self.assertEqual(unpack(out[16:],len(words)).tolist(),words.tolist())
    def test_state_is_not_current_observation(self):
        # +/-zero have equal numerical value, but current raw state roots differ.
        p,_=native_rounder();a=p.extract([0]*32);v=[0]*32;v[31]=1;b=p.extract(v)
        self.assertNotEqual(a[16:],b[16:])
    def test_fraction_extremes(self):
        self.assertEqual(fraction_round(0x7f7fffff),0x7f80)
        self.assertEqual(fraction_round(0xff7fffff),0xff80)
        self.assertEqual(fraction_round(0xffa00001),0x7fc0)
    def test_native_adjoint_order_witness(self):
        s=scope_run()['native_transpose'];self.assertEqual((s['balanced'],s['reverse_accumulation']),(1,2))
    def test_energy_unique_but_local_minimum(self):
        s=scope_run();states=s['energy_assignments']
        self.assertEqual([r['state'] for r in states if r['energy']==0],[[1,1,1]])
        self.assertTrue(all(e>1 for e in s['energy_local_minimum']['one_flip_energies']))
    def test_exact_direction_frame(self):
        s=scope_run()['directional'];self.assertEqual([r['rank'] for r in s['frame_ranks']],[1,2,3,4])
        self.assertNotEqual(s['target'],s['one_sample_reconstruction'])
    def test_matrix_cost_keeps_topology(self):
        p=transpose_binary_matrix([[1]*16 for _ in range(16)])
        self.assertGreater(p.costs()['serialized_bytes'],16*16//8)
if __name__=='__main__':unittest.main()
