import itertools,struct,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from routed import BDD,Program,compile_program,integer_to_bf16,BudgetExceeded,HEADER

def evaluate(bdd,root,bits,cache):
    if root<2:return root
    if root in cache:return cache[root]
    v,lo,hi=bdd.nodes[root]
    result=evaluate(bdd,hi if bits[v] else lo,bits,cache);cache[root]=result
    return result

def npword(v):
    bits=int(np.asarray(np.float32(v)).view(np.uint32))
    return ((bits+0x7fff+((bits>>16)&1))>>16)&65535

class Check(unittest.TestCase):
    def test_boolean_operations(self):
        b=BDD();x=b.mk(0,0,1);y=b.mk(1,0,1)
        for op,fn in [('&',lambda a,c:a&c),('|',lambda a,c:a|c),('^',lambda a,c:a^c)]:
            r=b.op(op,x,y)
            for a,c in itertools.product([0,1],repeat=2):self.assertEqual(evaluate(b,r,[a,c],{}),fn(a,c))
    def test_reduction_and_hash_consing(self):
        b=BDD();self.assertEqual(b.mk(0,1,1),1)
        self.assertEqual(b.mk(0,0,1),b.mk(0,0,1))
    def test_symbolic_rounding_all_13bit_integers(self):
        b=BDD();xs=[b.mk(j,0,1) for j in range(13)];roots=b.bf16(xs)
        for v in range(-4096,4096):
            bits=[(v>>j)&1 for j in range(13)];cache={}
            out=sum(evaluate(b,r,bits,cache)<<j for j,r in enumerate(roots))
            self.assertEqual(out,npword(v))
    def test_integer_reference(self):
        for v in range(-32768,32769):self.assertEqual(integer_to_bf16(v),npword(v))
    def test_round_ties_and_carry(self):
        for x in [0,1,-1,257,259,511,513,1022,1023,-257,-511]:
            self.assertEqual(integer_to_bf16(x),npword(x))
    def test_no_weights_at_query(self):
        w=[[3,-5],[7,1]];p,_=compile_program(w,4,'plane')
        w.clear()
        self.assertEqual(p.run([2,3])[0],[npword(-9),npword(17)])
        self.assertFalse(hasattr(p,'weights'))
    def test_orders_exhaustive(self):
        w=[[127,-113],[59,67]]
        p,_=compile_program(w,4,'input');q,_=compile_program(w,4,'plane')
        for x in itertools.product(range(-8,8),repeat=2):
            ref=[npword(sum(a*b for a,b in zip(row,x))) for row in w]
            self.assertEqual(p.run(list(x))[0],ref)
            self.assertEqual(q.run(list(x))[0],ref)
    def test_zero_cancellation(self):
        p,_=compile_program([[1,-1]],4,'plane')
        for x in range(-8,8):self.assertEqual(p.run([x,x])[0],[0])
    def test_roundtrip_and_accounting(self):
        p,s=compile_program([[3,7]],4,'plane');q=Program(p.data)
        a,c=q.run([7,-8]);self.assertEqual(a,[npword(-35)])
        self.assertEqual(c['model_code_bytes'],12*c['visited_nodes']+c['root_payload_bytes']+c['varmap_bytes'])
        self.assertEqual(c['memo_inserts'],c['visited_nodes'])
        self.assertEqual(s['constructor_coefficient_visits'],6)
        self.assertEqual(s['program_bytes'],len(p.data))
        self.assertEqual(s['input_truth_table_rows_built'],0)
    def test_budget_refusal(self):
        with self.assertRaises(BudgetExceeded):compile_program([[3,7]],4,'plane',node_cap=10)
    def test_bad_inputs(self):
        p,_=compile_program([[1]],4,'plane')
        for x in [[8],[-9],[1.0],[],[1,2]]:
            with self.assertRaises(ValueError):p.run(x)
    def test_bad_weights(self):
        for w in [[],[[]],[[128]],[[1,2],[3]],[[1.5]]]:
            with self.assertRaises(ValueError):compile_program(w,4,'plane')
    def test_bad_program(self):
        p,_=compile_program([[1]],4,'plane')
        for data in [b'',p.data[:-1],b'BAD!'+p.data[4:],p.data+b'X']:
            with self.assertRaises(ValueError):Program(data)
    def test_no_whole_graph_memo_reset(self):
        p,_=compile_program([[7,-5,3]],4,'plane')
        _,a=p.run([0,0,0]);_,b=p.run([1,2,3])
        self.assertLess(a['memo_inserts'],p.num_nodes)
        self.assertLess(b['memo_inserts'],p.num_nodes)
    def test_deterministic_serialization(self):
        w=[[1,-3,5],[3,1,7]]
        a,_=compile_program(w,4,'plane');b,_=compile_program(w,4,'plane')
        self.assertEqual(a.data,b.data)
if __name__=='__main__':unittest.main()
