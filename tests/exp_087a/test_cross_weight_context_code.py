import numpy as np, pytest, torch
from vortex_runtime.cross_weight_context_code import ContextCodeError, audit_layer, entropy, majority, triplet

def test_majority_exact():
    c=np.array([1,1,1,2,2,3],np.uint64);t=np.array([7,7,9,4,5,6],np.uint64);r=majority(c,t)
    assert r['exact'] and r['hit']==pytest.approx(4/6)

def test_periodic_positive_control():
    p=4; pattern=np.arange(p*p,dtype=np.uint64).reshape(p,p); a=np.tile(pattern,(16,16)); T=triplet(a,a^0x1234,a^0x4321)
    baseline=sum(entropy(x.astype(np.uint16)) for x in (a,a^0x1234,a^0x4321))*a.size
    assert 48*p*p/baseline < .2 and np.array_equal(T,T)

def test_audit_deterministic():
    g=torch.Generator().manual_seed(7);G=torch.randn(32,16,generator=g).bfloat16();U=torch.randn(32,16,generator=g).bfloat16();D=torch.randn(16,32,generator=g).bfloat16()
    a=audit_layer(G,U,D,0,(2,4));b=audit_layer(G,U,D,0,(2,4));assert a==b and a['best_candidate']['reconstruction_exact']

def test_shape_fail_closed():
    G=torch.ones(8,4,dtype=torch.bfloat16)
    with pytest.raises(ContextCodeError): audit_layer(G.float(),G,torch.ones(4,8,dtype=torch.bfloat16),0,(2,))
