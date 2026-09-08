"""Independent scalar decoder/reference. Does not import either source module."""
import hashlib
import json
import struct
from pathlib import Path

def f(p, terms):
    answer=0
    for m in terms:
        v=1
        while m:
            bit=m & -m
            v &= bool(p & bit)
            m-=bit
        answer ^= int(v)
    return answer

def verify(root):
    cases={}
    for path in root.glob('*.bin'):
        fixture=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        data=path.read_bytes(); body=data[:-32]
        assert hashlib.sha256(body).digest()==data[-32:]
        magic,n,nf,rank,k=struct.unpack_from('<8sIIII',body)
        assert magic==b'VXCS01\0\0' and n==len(fixture['words'])
        assert nf==len(fixture['functions']) and k==fixture['k']
        nb=(n+7)//8; rb=(rank+7)//8; start=24+nf*(1+rb)
        assert len(body)==start+rank*nb
        constants=list(body[24:24+nf])
        masks=[int.from_bytes(body[24+nf+i*rb:24+nf+(i+1)*rb],'little') for i in range(nf)]
        planes=[int.from_bytes(body[start+i*nb:start+(i+1)*nb],'little') for i in range(rank)]
        # Verify pointwise represented functions on every SOURCE position.
        for i,p in enumerate(fixture['words']):
            for j,fn in enumerate(fixture['functions']):
                a=constants[j]
                for l in range(rank):
                    if masks[j]>>l&1:
                        a ^= planes[l]>>i&1
                assert a==f(p,fn)
        cases[path.stem]=(fixture,constants,masks,planes)
    queries=bits=0; seen=set()
    for line in (root/'raw.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line); key=(r['case'],r['query']); assert key not in seen;seen.add(key)
        fixture,constants,masks,planes=cases[r['case']]
        q=fixture['masks'][r['query']]; answers=[]
        for j,fn in enumerate(fixture['functions']):
            a=constants[j]*(q.bit_count()%2)
            for l in range(len(planes)):
                if masks[j]>>l&1:
                    # Intentionally bit-by-bit, not the candidate word loop.
                    a ^= sum(((q>>i)&1)*((planes[l]>>i)&1) for i in range(len(fixture['words'])))%2
            direct=sum(f(p,fn) for i,p in enumerate(fixture['words']) if q>>i&1)%2
            assert a==direct
            answers.append(a)
        assert answers==r['actual']==r['expected']
        queries+=1; bits+=len(answers)
    assert queries==sum(len(v[0]['masks']) for v in cases.values())
    s=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    assert (queries,bits)==(s['queries'],s['output_bits'])
    return dict(cases=len(cases),queries=queries,output_bits=bits,mismatches=0,
                independent_parser=True,independent_scalar_boolean_reference=True,
                native_rng_state_validation='NOT TESTED')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(verify(a.results),indent=2,sort_keys=True)+'\n',encoding='utf-8')
