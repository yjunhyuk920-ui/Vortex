"""Independent arithmetic checks and scope/accounting guards."""
from pathlib import Path
from fractions import Fraction as Q
import base64, hashlib, importlib.util, json, math, struct, sys, unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import research as r

def pow2(e):return Q(2**e) if e>=0 else Q(1,2**-e)
def rne(v,p):
    """Independent rational RNE for finite normal-range test values only."""
    v=Q(v)
    if not v:return Q(0)
    sign=1 if v>0 else -1; a=abs(v)
    e=a.numerator.bit_length()-a.denominator.bit_length()
    if a<pow2(e):e-=1
    step=pow2(e-p+1);u=a/step;k,rem=divmod(u.numerator,u.denominator)
    k+= int(2*rem>u.denominator or (2*rem==u.denominator and (k&1)))
    return sign*k*step

class TestResearch(unittest.TestCase):
    def test_exact_zero_quadratic_nonzero_output(self):
        s=json.loads((ROOT/'results/summary.json').read_text())['zero_quadratic_nonzero_output']
        self.assertTrue(s['exact_zero_quadratic']);self.assertTrue(s['all_weight_entries_nonzero'])
        for field in ['G','U']:
            a=s[field];self.assertNotEqual(Q(a[0][0])*Q(a[1][1])-Q(a[0][1])*Q(a[1][0]),0)
        self.assertEqual(s['native_output'],[7.875])
    def test_independent_attention_key_rounding(self):
        self.assertEqual(rne(257,8),256);self.assertEqual(rne(513,8),512)
        native=2*rne(257,8)-rne(513,8)
        moved=(2-1)*1+(2-2)*256
        self.assertEqual(native,0);self.assertEqual(moved,1)
    def test_independent_accumulation_rounding(self):
        h=Q(2**24)
        native=rne(rne(h+1,24)+rne(-h+1,24),24)
        self.assertEqual(native,1);self.assertEqual(h+1-h+1,2)
    def test_projection_cast_counterexample(self):
        self.assertEqual(rne(Q(257,256),8),1)
    def test_actual_packed_program_roundtrip(self):
        for item in json.loads((ROOT/'results/serialized_programs.json').read_text()):
            b=base64.b64decode(item['base64'])
            self.assertEqual(len(b),item['bytes']);self.assertEqual(hashlib.sha256(b).hexdigest(),item['sha256'])
            p=r.deserialize_monomials(b);self.assertEqual(len(p['pos']),16)
            self.assertEqual(len(p['shifts']),8)
    def test_packed_program_rejects_corruption(self):
        p=r.compile_monomials(r.bf([[1,-2],[3,4]]));b=r.serialize_monomials(p)
        for bad in [b[:10],b'FAIL'+b[4:],b[:-1],b+b'x']:
            with self.assertRaises(ValueError):r.deserialize_monomials(bad)
    def test_monomial_real_projection_on_dyadic_example(self):
        w=r.bf([[1,.25],[-.5,2]]);x=r.bf([.5,-.25]);p=r.compile_monomials(w)
        y=r.monomial_projection(p,x.float().numpy())
        self.assertLess(max(abs(y[0]-.4375),abs(y[1]+.75)),1e-12)
    def test_exact_quadratic_compiler(self):
        g=[[Q(1),Q(2)],[Q(3),Q(4)]];u=[[Q(2),Q(-1)],[Q(1),Q(2)]];d=[[Q(2),Q(-3)]]
        pairs,c=r.compile_quadratic(g,u,d)
        for x in [[Q(2),Q(3)],[Q(1,4),Q(-5,2)]]:
            self.assertEqual(r.eval_quadratic(pairs,c,x),r.direct_bilinear(g,u,d,x))
    def test_scope_not_promoted_after_sample_pass(self):
        s=json.loads((ROOT/'results/summary.json').read_text());m=s['mission']
        self.assertFalse(m['CORE_ADMISSION']);self.assertEqual(m['O1_O6'],'OPEN')
        self.assertFalse(m['three_qualifying_new_principles_constructed'])
        self.assertFalse(m['public_checkpoint_tested']);self.assertEqual(m['HARDWARE_STATUS'],'NOT_TESTED')
        self.assertGreater(s['native_parity_rewrite_mismatches'],0)
        self.assertGreater(s['monomial_direct_mismatches'],0)
    def test_preregistration_unchanged(self):
        digest=hashlib.sha256((ROOT/'PREREGISTRATION.json').read_bytes()).hexdigest()
        self.assertEqual(digest,'90e5da2d8b06f07d057f7d4fb0b86dbf010ee0f9e12b000718907947ea6c5679')
    def test_full_tensor_scale_already_exceeds_memory(self):
        s=r.tensor_scale();self.assertEqual(s['optimistic_two_bytes_per_tensor_coefficient_GiB'],4096.25)
        self.assertEqual(s['original_BF16_projection_payload_GiB'],4.875)
        self.assertEqual(s['odd_direct_branch_alone_vs_original_projection_reads'],1)
    def test_no_fast_monomial_count_claim(self):
        s=json.loads((ROOT/'results/summary.json').read_text())
        for p in s['monomial_programs']:
            self.assertGreater(p['packed_program_bytes'],p['original_weight_bytes'])
            self.assertGreater(p['binary_power_and_product_multiplications'],p['reference_projection_multiplications'])

if __name__=='__main__':unittest.main(verbosity=2)
