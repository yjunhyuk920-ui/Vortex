from __future__ import annotations
import hashlib,json,math,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from native_transfer import *
ROOT=Path(__file__).resolve().parents[2]

def run():
    confpath=ROOT/'experiments/native_transfer/continuation.json'
    conf=json.loads(confpath.read_text()); rng=random.Random(conf['tests']['seed'])
    mismatch=cases=direct=zero=0
    rows=[]
    for i in range(conf['tests']['random_finite_cases']):
        ops=[]
        for j in range(3):
            u=rng.getrandbits(32)
            if u&0x7F800000==0x7F800000:u^=1<<23
            ops.append(value(u))
        w,x,a=ops; target=fma(w,x,a)
        if not math.isfinite(target):continue
        rank=rank_word(bits(target))
        # Mix exact word and multiword intervals; include cancellation/subnormal cells.
        pad=rng.randrange(0,5)
        lo=max(MIN_FINITE_RANK,rank-pad); hi=min(MAX_FINITE_RANK,rank+pad)
        slow=preimage(w,x,lo,hi); fast=preimage_direct(w,x,lo,hi)
        same=(None if slow is None else slow[:2])==(None if fast is None else fast[:2])
        mismatch+=not same; cases+=1
        if fast is not None:
            direct+=fast[2]=='exact_rounding_cell'; zero+=fast[2]=='native_zero_boundary'
        rows.append({'case':i,'input_words':[bits(v) for v in ops],'output_ranks':[lo,hi],
                     'native_search':slow,'direct_inverse':fast,'equal':same})
    # Deterministic all signed-zero endpoint combinations and extreme cells.
    for w,x in [(0.0,1.0),(-0.0,1.0),(1.0,1.0),(-1.0,1.0),(value(1),0.5)]:
        for ow in [0,0x80000000,1,0x80000001,0x7F7FFFFF,0xFF7FFFFF,0x3F800000]:
            rank=rank_word(ow); slow=preimage(w,x,rank,rank); fast=preimage_direct(w,x,rank,rank)
            same=(None if slow is None else slow[:2])==(None if fast is None else fast[:2])
            mismatch+=not same;cases+=1
            if fast is not None:
                direct+=fast[2]=='exact_rounding_cell';zero+=fast[2]=='native_zero_boundary'
            rows.append({'case':'boundary','input_words':[bits(w),bits(x)],'output_word':ow,
                         'native_search':slow,'direct_inverse':fast,'equal':same})
    suffix_mismatch=0; suffix_rows=[]
    for i in range(conf['tests']['suffix_inverse_cases']):
        n=rng.randrange(1,17); w=[value(bits(rng.randrange(-64,65)/8)) for _ in range(n)]
        x=[value(bits(rng.randrange(-64,65)/8)) for _ in range(n)]
        initial=value(bits(rng.randrange(-1024,1025)/16))
        output=reference_dot(w,x,initial); ans=suffix_preimage(w,x,bits(output))
        ok=ans is not None and ans[0]<=rank_word(bits(initial))<=ans[1]
        if ans is not None:
            for rank in (ans[0],ans[1],(ans[0]+ans[1])//2):
                ok=ok and bits(reference_dot(w,x,value(unrank_word(rank))))==bits(output)
        suffix_mismatch+=not ok
        suffix_rows.append({'case':i,'n':n,'w_words':[bits(t) for t in w],'x_words':[bits(t) for t in x],
                            'initial_word':bits(initial),'output_word':bits(output),'inverse':ans,'checked':ok})
    return {'version':conf['version'],'continuation_sha256':hashlib.sha256(confpath.read_bytes()).hexdigest(),
            'interval_cases':cases,'interval_mismatches':mismatch,'nonempty_direct_intervals':direct,
            'nonempty_native_zero_intervals':zero,'suffix_cases':len(suffix_rows),'suffix_mismatches':suffix_mismatch,
            'scope':'Exact native inverse primitive; known test outputs are oracle controls, not a deployment selector.',
            'cost':'Nonzero-boundary inverse uses bounded exact dyadic arithmetic and no native search. Zero boundaries retain <=64 native calls/stage. Products and verification remain charged.',
            'full_mission_status':'NOT_ESTABLISHED'},rows,suffix_rows

if __name__=='__main__':
    summary,rows,suffix=run(); out=ROOT/'results/native_transfer'
    for name,data in [('inverse_summary.json',summary),('inverse_rows.json',rows),('inverse_suffix_rows.json',suffix)]:
        (out/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
