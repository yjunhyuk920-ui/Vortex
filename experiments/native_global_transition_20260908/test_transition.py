import json,operator,tempfile,unittest
from pathlib import Path
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from transformers import DynamicCache
from native_graph import (raw_bytes,tensor_meta,cache_flat,native_cache,pure_cse_dce,
    proof_guards,work_inventory,export_program,load_program,DemandGraphExecutor,encode,decode)
from native_state import pack_cache,unpack_cache,PackedCache


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.arange(6,dtype=torch.float32).reshape(2,3))
    def forward(self,x):
        wt=self.weight.t();a=x@wt;b=x@wt;unused=x-x
        return a+b,x.clone()


class Tests(unittest.TestCase):
    def setUp(self):self.x=torch.tensor([[1.0,2.0,3.0]])

    def test_01_common_subexpression_really_eliminates_work_in_control(self):
        m=Toy();g=make_fx(m)(self.x);before=work_inventory(g)
        r=pure_cse_dce(g);after=work_inventory(g)
        self.assertTrue(r['cse']);self.assertLess(after['matrix_macs'],before['matrix_macs'])
        self.assertEqual([raw_bytes(t) for t in m(self.x)],[raw_bytes(t) for t in g(self.x)])

    def test_02_distinct_external_allocations_are_not_merged(self):
        g=make_fx(lambda x:(x+1,x+1))(self.x);r=pure_cse_dce(g)
        a,b=g(self.x);self.assertNotEqual(a.data_ptr(),b.data_ptr());self.assertFalse(r['cse'])

    def test_03_mutation_rejected(self):
        g=make_fx(lambda x:x.add_(1))(self.x.clone())
        with self.assertRaises(ValueError):proof_guards(g)

    def test_04_random_kernel_rejected(self):
        g=make_fx(lambda x:x+torch.rand_like(x))(self.x)
        with self.assertRaises(ValueError):proof_guards(g)

    def test_05_checkpoint_bound_program_reload(self):
        m=Toy();g=make_fx(m)(self.x);pure_cse_dce(g)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'p.json';d=export_program(g,m,p);loaded=load_program(p,m)
            self.assertFalse(d['model_weights_embedded'])
            self.assertTrue(any(v['kind']=='checkpoint' for v in d['attributes'].values()))
            for x in [self.x,-self.x,self.x*0,self.x*13]:
                self.assertEqual([raw_bytes(t) for t in m(x)],[raw_bytes(t) for t in loaded(x)])

    def test_06_format_fault_rejected(self):
        m=Toy();g=make_fx(m)(self.x)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'p.json';d=export_program(g,m,p);d['format']='bad';p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):load_program(p,m)

    def test_07_demand_interpreter_no_desired_answers(self):
        m=Toy();g=make_fx(m)(self.x);pure_cse_dce(g)
        e=object.__new__(DemandGraphExecutor);e.demand_records=[]
        self.assertEqual([raw_bytes(t) for t in m(self.x)],[raw_bytes(t) for t in e.execute(g,(self.x,))])
        self.assertEqual(e.demand_records[0]['supplied_desired_output_bits'],0)
        self.assertEqual(e.demand_records[0]['backward_narrowings'],0)

    def test_08_packed_kv_layout_and_bits(self):
        c=DynamicCache()
        for i in range(2):
            k=torch.arange(768,dtype=torch.float32).to(torch.bfloat16).reshape(1,4,3,64).transpose(1,2)
            v=-k.clone();c.update(k,v,i)
        p=pack_cache(c);d=unpack_cache(p)
        self.assertEqual(p.payload.numel(),sum(t.numel()*t.element_size() for t in cache_flat(c)))
        self.assertEqual(p.get_seq_length(),4)
        for a,b in zip(cache_flat(c),cache_flat(d)):
            self.assertEqual(raw_bytes(a),raw_bytes(b));self.assertEqual(tensor_meta(a),tensor_meta(b))

    def test_09_offset_and_negative_zero_payload(self):
        c=DynamicCache();data=torch.arange(1024,dtype=torch.float32).to(torch.bfloat16)
        k=data[64:448].reshape(1,3,2,64);k.flatten()[0]=-0.0
        v=k.clone();c.update(k,v,0);d=unpack_cache(pack_cache(c))
        for a,b in zip(cache_flat(c),cache_flat(d)):
            self.assertEqual(raw_bytes(a),raw_bytes(b));self.assertEqual(tensor_meta(a),tensor_meta(b))

    def test_10_packed_has_no_hidden_raw_kv_layers(self):
        c=DynamicCache();x=torch.ones((1,3,2,64),dtype=torch.bfloat16);c.update(x,x.clone(),0)
        p=pack_cache(c)
        self.assertTrue(all(layer.keys is None and layer.values is None for layer in p.layers))

    def test_11_empty_cache(self):
        c=DynamicCache();p=pack_cache(c);d=unpack_cache(p)
        self.assertEqual(p.payload.numel(),0);self.assertEqual(d.get_seq_length(),0)

    def test_12_literals_roundtrip_signed_zero(self):
        obj=(slice(1,None,2),torch.float32,torch.device('cpu'),-0.0,[1,True,None])
        back=decode(encode(obj),{})
        self.assertEqual(back[3].hex(),'-0x0.0p+0');self.assertEqual(back[1],torch.float32)

    def test_13_cache_adapter_roundtrip(self):
        x=torch.arange(384,dtype=torch.float32).to(torch.bfloat16).reshape(1,3,2,64)
        c=native_cache((x,-x));self.assertEqual(raw_bytes(cache_flat(c)[0]),raw_bytes(x))

    def test_14_live_state_root_is_retained(self):
        g=make_fx(lambda x:(torch.zeros_like(x),x*3))(self.x);pure_cse_dce(g)
        y,s=g(self.x);self.assertTrue(torch.equal(s,self.x*3))
        self.assertEqual(len(work_inventory(g)['root_names']),2)


if __name__=='__main__':unittest.main(verbosity=2)
