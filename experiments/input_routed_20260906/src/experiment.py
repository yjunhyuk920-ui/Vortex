from __future__ import annotations
import argparse,hashlib,itertools,json,random,sys,time,platform,resource
from pathlib import Path
import numpy as np
from routed import compile_program,BudgetExceeded,integer_to_bf16,Program

CONFIG=[(2,4,7),(4,4,7),(4,16,7),(8,4,7),(8,16,7),(16,4,7),(4,4,127),(8,16,127)]
SEEDS=[271,811]
ORDERS=['input','plane']


def dump(path,obj):
    path.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')


def frozen_cases():
    cases=[]
    for case_id,(n,m,maxw) in enumerate(CONFIG):
        for seed in SEEDS:
            rng=random.Random(seed+10000*case_id)
            pool=[a for a in range(-maxw,maxw+1) if a]
            w=[[rng.choice(pool) for _ in range(n)] for _ in range(m)]
            if n==2: queries=[list(x) for x in itertools.product(range(-8,8),repeat=n)]
            else:
                q=random.Random(seed+800_000+case_id)
                queries=[[q.randrange(-8,8) for _ in range(n)] for _ in range(64)]
                queries += [[-8]*n,[7]*n,[0]*n,[(-8 if j%2==0 else 7) for j in range(n)]]
            cases.append(dict(case_id=case_id,seed=seed,n=n,m=m,maxw=maxw,p=4,w=w,queries=queries))
    return cases


def native(w,x):
    y=[]
    for row in w:
        acc=np.float32(0)
        for a,b in zip(row,x): acc=np.float32(acc+np.float32(np.float32(a)*np.float32(b)))
        bits=int(np.asarray(acc,dtype=np.float32).view(np.uint32))
        y.append(((bits+0x7fff+((bits>>16)&1))>>16)&0xffff)
    return y


def quantiles(xs):
    a=sorted(xs)
    return dict(min=a[0],median=a[len(a)//2],p95=a[int(.95*(len(a)-1))],max=a[-1])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cases=frozen_cases();dump(out/'inputs.json',cases)
    input_sha=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest()
    dump(out/'frozen.json',dict(input_sha256=input_sha,config=CONFIG,seeds=SEEDS,orders=ORDERS,
                              node_cap=100000,cache_cap=300000))
    records=[];raw=[];timings=[]
    for case in cases:
        for order in ORDERS:
            tag=f"c{case['case_id']}_s{case['seed']}_{order}"
            before=time.perf_counter()
            try:
                program,stats=compile_program(case['w'],4,order)
            except BudgetExceeded as e:
                elapsed=time.perf_counter()-before
                record=dict(tag=tag,n=case['n'],m=case['m'],maxw=case['maxw'],order=order,
                            status='REFUSED_CONSTRUCTOR_CAP',reason=str(e),**e.stats)
                records.append(record);print(tag,record['status'],record['constructed_nodes'],flush=True)
                timings.append(dict(tag=tag,construction_seconds=elapsed));continue
            elapsed=time.perf_counter()-before
            path=out/(tag+'.vrc');path.write_bytes(program.data)
            program=Program(path.read_bytes())
            samples=[];mismatch=0
            for qi,x in enumerate(case['queries']):
                y,counters=program.run(x)
                r=native(case['w'],x)
                exact=[integer_to_bf16(sum(a*b for a,b in zip(row,x))) for row in case['w']]
                mismatch += sum(a!=b or a!=c for a,b,c in zip(y,r,exact))
                samples.append(counters)
                raw.append(dict(tag=tag,query_index=qi,x=x,y=y,native=r,integer_ref=exact,counters=counters))
            base=stats['original_weight_bytes']
            record=dict(tag=tag,n=case['n'],m=case['m'],maxw=case['maxw'],order=order,
                        status='COMPLETE',**stats,program_sha256=hashlib.sha256(program.data).hexdigest(),
                        queries=len(samples),output_words=len(samples)*case['m'],mismatches=mismatch,
                        visited_nodes=quantiles([s['visited_nodes'] for s in samples]),
                        code_read_ratio=quantiles([s['model_code_bytes']/base for s in samples]),
                        program_size_ratio=len(program.data)/base,
                        code_fraction_visited=quantiles([s['node_payload_bytes']/len(program.data) for s in samples]),
                        logical64_ratio=quantiles([64*s['unique_64byte_blocks']/base for s in samples]),
                        logical4096_ratio=quantiles([4096*s['unique_4096byte_pages']/base for s in samples]),
                        input_truth_table_constructed=False)
            records.append(record);timings.append(dict(tag=tag,construction_seconds=elapsed))
            print(tag,'COMPLETE',stats['live_nodes'],'median ratio',record['code_read_ratio']['median'],flush=True)
    completed=[r for r in records if r['status']=='COMPLETE']
    summary=dict(input_sha256=input_sha,attempts=len(records),completed=len(completed),
                 refused=len(records)-len(completed),mismatches=sum(r['mismatches'] for r in completed),
                 query_count=sum(r['queries'] for r in completed),
                 output_words=sum(r['output_words'] for r in completed),
                 all_registered_gate_pass=False,theory_status='NOT_ESTABLISHED',core_admission=False,
                 hardware_status='NOT_TESTED',rows=records)
    dump(out/'summary.json',summary)
    with (out/'raw.jsonl').open('w') as f:
        for row in raw:f.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
    dump(out/'measurements.json',dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
                                    process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                    construction_times=timings,
                                    note='local CPU observations; non-deterministic; not inference benchmark'))
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)
    if summary['mismatches']: raise SystemExit('MISMATCH')

if __name__=='__main__':main()
