import itertools,struct,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from residue import *
class Tests(unittest.TestCase):
 def test_interval_exhaustive(self):
  for q in (1,2,4,8,16):
   for lo in range(-12,13):
    for width in range(q):
     hi=lo+width
     for y in range(lo,hi+1):self.assertEqual(lift_interval(y%q,q,lo,hi),y)
 def test_ambiguous(self):
  with self.assertRaises(ValueError):lift_interval(0,4,0,4)
 def test_absent(self):
  with self.assertRaises(ValueError):lift_interval(3,4,0,2)
 def test_negative(self):self.assertEqual(lift_interval(13,16,-5,5),-3)
 def test_dense_corrections(self):
  w=[[1,-2,3,4],[2,4,-3,1],[1,5,2,-4],[4,-1,6,1]]
  blob,_=compile_matrix(w);pr=Program(blob)
  for x in itertools.product((-1,0,1),repeat=4):
   got,log=pr.run(list(x));self.assertEqual(got,[sum(a*b for a,b in zip(r,x)) for r in w])
 def test_file(self):
  b,d=compile_matrix([[1,2,3,4]]);self.assertEqual(Program(bytes(b)).run([1,0,-1,1])[0],[2])
 def test_wrong_input(self):
  b,_=compile_matrix([[1,2,3,4]])
  with self.assertRaises(ValueError):Program(b).run([1,2,1,1])
 def test_no_weight_conversion(self):
  with self.assertRaises(ValueError):compile_matrix([[1, 1.5]])
 def test_fractional_input(self):
  b,_=compile_matrix([[1,2]])
  with self.assertRaises(ValueError):Program(b).run([1,0.5])
 def test_range(self):
  with self.assertRaises(ValueError):compile_matrix([[1,128]])
 def test_zero_sign_domain(self):
  with self.assertRaises(ValueError):compile_matrix([[-1,-1]])
 def test_exact_int_bf16(self):
  self.assertEqual(bf16_bits_int(1),0x3f80);self.assertEqual(bf16_bits_int(-1),0xbf80)
  self.assertEqual(bf16_bits_int(257),0x4380);self.assertEqual(bf16_bits_int(259),0x4382)
 def test_zero(self):
  b,_=compile_matrix([[1,-2,3,4]])
  self.assertEqual(Program(b).run([0,0,0,0])[0],[0])
 def test_constant(self):
  b,d=compile_matrix([[3,3,3,3]])
  self.assertEqual(d['rows'][0]['stored_signed_bits'],0)
  self.assertEqual(Program(b).run([1,-1,1,0])[0],[3])
 def test_format(self):
  with self.assertRaises(ValueError):Program(b'bad')
 def test_coeff_recoverable(self):
  for n in range(1,9):
   r=[(-1)**j*(j+1) for j in range(n)];B=sum(map(abs,r));q=1<<(2*B).bit_length()
   self.assertTrue(all(lift_interval(v%q,q,-B,B)==v for v in r))
if __name__=='__main__':unittest.main()
