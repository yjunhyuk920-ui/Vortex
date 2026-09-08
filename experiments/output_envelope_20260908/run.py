from __future__ import annotations
import argparse,datetime,hashlib,json,platform,struct,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from envelope import Plan,execute,direct,encode
import oracle
PIN='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
MODEL='HuggingFaceTB/SmolLM2-135M'
ROLES=['self_attn.q_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
NAMES=[f'model.layers.{l}.{r}.weight' for l in [0,15,29] for r in ROLES]
ROOT=Path(__file__).resolve().parent

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(b):return hashlib.sha256(b).hexdigest()

class Acquisition:
    def __init__(self,cache):
        self.cache=cache;cache.mkdir(parents=True,exist_ok=True);self.receipts=[]
    def ranged(self,start,end):
        import requests
        url=f'https://huggingface.co/{MODEL}/resolve/{PIN}/model.safetensors?download=true&range_start={start}&range_end={end}'
        r=requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(15,90),stream=True)
        expected=end-start+1;cr=r.headers.get('Content-Range','')
        if r.status_code!=206 or not cr.startswith(f'bytes {start}-{end}/'):
            r.close();raise RuntimeError(f'Range not honored: status={r.status_code}, range={cr}')
        parts=[];count=0
        for p in r.iter_content(1<<20):
            count+=len(p)
            if count>expected:r.close();raise RuntimeError('oversized range response')
            parts.append(p)
        data=b''.join(parts)
        if len(data)!=expected:raise RuntimeError('truncated response')
        self.receipts.append({'start':start,'end':end,'status':r.status_code,'content_range':cr,'bytes':len(data),'sha256':sha(data)})
        # Preserve receipts after every acquired range, even if a later read fails.
        dump(self.cache/'acquisition_receipts.json',self.receipts)
        return data
    def header(self):
        p=self.cache/'safetensors_header.json'
        if p.exists():return json.loads(p.read_text(encoding='utf-8'))
        n=struct.unpack('<Q',self.ranged(0,7))[0]
        if n>10_000_000:raise RuntimeError('header too large')
        raw=self.ranged(8,7+n)
        d={'header_bytes':n,'header':json.loads(raw),'raw_header_sha256':sha(raw),'pin':PIN,'model':MODEL}
        dump(p,d);return d
    def tensor(self,name):
        header=self.header();t=header['header'][name]
        if t['dtype']!='BF16' or len(t['shape'])!=2:raise RuntimeError('unmodified BF16 matrix required')
        lo,hi=t['data_offsets'];m,n=t['shape']
        if hi-lo!=2*m*n:raise RuntimeError('invalid tensor byte count')
        p=self.cache/(name+'.u16')
        if not p.exists():
            base=8+header['header_bytes'];p.write_bytes(self.ranged(base+lo,base+hi-1))
        data=p.read_bytes()
        if len(data)!=hi-lo:raise RuntimeError('cache length mismatch')
        return p,np.frombuffer(data,dtype='<u2').copy().reshape(m,n),{'name':name,'shape':[m,n],'dtype':'BF16','bytes':len(data),'sha256':sha(data),'data_offsets':t['data_offsets']}

def run(output,cache):
    output.mkdir(parents=True,exist_ok=False);acq=Acquisition(cache)
    frozen={str(p.relative_to(ROOT)).replace('\\','/'):sha(p.read_bytes()) for p in sorted(ROOT.rglob('*.py')) if not any(x in p.parts for x in ['.cache','replay','__pycache__'])}
    frozen['PREREGISTRATION.md']=sha((ROOT/'PREREGISTRATION.md').read_bytes())
    dump(output/'source_before_payload.json',frozen)
    rows=[];acquisitions=[];runtime=[]
    for name in NAMES:
        tick=time.perf_counter();path,w,receipt=acq.tensor(name);acquisitions.append(receipt)
        plan=Plan.build(w,256);planpath=output/'plans'/(name+'.npz');planpath.parent.mkdir(exist_ok=True)
        plan.save(planpath);plan=Plan.load(planpath);prep=time.perf_counter()-tick;m,n=w.shape
        rng=np.random.default_rng(260908);inputs=encode(rng.normal(size=(8,n)))
        np.save(output/(name+'.inputs.npy'),inputs,allow_pickle=False)
        details=[];mismatches=0;fracchecks=0
        for k,x in enumerate(inputs):
            reads=[]
            def reader(a,b):
                with path.open('rb') as f:
                    f.seek(a*n*2);raw=f.read((b-a)*n*2)
                if len(raw)!=(b-a)*n*2:raise RuntimeError('short cold read')
                reads.append({'first_row':a,'end_row':b,'bytes':len(raw)})
                return np.frombuffer(raw,dtype='<u2').reshape(b-a,n)
            y,stats=execute(plan,x,reader);ref=direct(w,x)
            bad=int(np.count_nonzero(y!=ref));mismatches+=bad
            if k==0:
                for i in range(min(3,m)):
                    test=oracle.dot(w[i],x)
                    if test!=int(y[i]):raise AssertionError(('Fraction mismatch',name,i,test,int(y[i])))
                    fracchecks+=1
            eligible=0;unique=[]
            for a in range(0,m,256):
                vals=ref[a:a+256];num=len(np.unique(vals));unique.append(num)
                eligible+=int(num==1 and (int(vals[0])&0x7fff)!=0 and (int(vals[0])&0x7f80)!=0x7f80)
            if stats['accepted_groups']>eligible:raise AssertionError('impossible certification')
            if sum(v['bytes'] for v in reads)!=stats['cold_weight_bytes_read']:raise AssertionError('traffic mismatch')
            stats.update({'input_index':k,'bf16_mismatches':bad,'oracle_broadcastable_groups':eligible,
                          'distinct_output_words_per_group':unique,'actual_cold_read_calls':reads,
                          'reference_output_u16':[int(v) for v in ref],
                          'candidate_output_u16':[int(v) for v in y]})
            details.append(stats)
        if mismatches:raise AssertionError(('output mismatches',name,mismatches))
        result={'tensor':receipt,'source_plan_file_bytes':planpath.stat().st_size,
                'input_sha256':sha(inputs.astype('<u2').tobytes()),'queries':details,
                'fraction_checks':fracchecks,'mismatches':mismatches}
        dump(output/(name+'.json'),result)
        rows.append({'name':name,'shape':[m,n],'queries':8,'coordinates':8*m,
                     'groups':sum(d['groups'] for d in details),
                     'accepted_groups':sum(d['accepted_groups'] for d in details),
                     'accepted_rows':sum(d['accepted_rows'] for d in details),
                     'oracle_broadcastable_groups':sum(d['oracle_broadcastable_groups'] for d in details),
                     'mismatches':mismatches,'fraction_checks':fracchecks,
                     'raw_weight_bytes':m*n*2,'summary_payload_bytes':plan.low.nbytes+plan.high.nbytes,
                     'source_plan_file_bytes':planpath.stat().st_size,
                     'cold_weight_fraction':sum(d['cold_weight_bytes_read'] for d in details)/(8*m*n*2),
                     'coefficient_payload_ratio':sum(d['coefficient_payload_ratio'] for d in details)/8,
                     'fp_work_ratio':sum(d['fp_work_ratio'] for d in details)/8,
                     'min_output_diversity_per_group':min(min(d['distinct_output_words_per_group']) for d in details)})
        runtime.append({'name':name,'acquire_and_build_seconds_observed':prep})
        print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    rr=np.random.default_rng(7);w=encode(rr.normal(size=(256,64))*2**-22);w[:,0]=encode(np.ones(256))
    x=encode(np.ones(64))
    def no_read(*a):raise AssertionError('positive control cold read')
    y,positive=execute(Plan.build(w),x,no_read);np.testing.assert_array_equal(y,direct(w,x));positive.pop('groups_trace')
    raw=sum(r['raw_weight_bytes'] for r in rows)
    summary={'model':MODEL,'revision':PIN,'activation_source':'SYNTHETIC_NOT_HF_FORWARD',
             'weight_mutation':False,'packet':256,'matrices':len(rows),
             'queries':sum(r['queries'] for r in rows),'output_coordinates':sum(r['coordinates'] for r in rows),
             'groups':sum(r['groups'] for r in rows),'accepted_groups':sum(r['accepted_groups'] for r in rows),
             'accepted_rows':sum(r['accepted_rows'] for r in rows),
             'oracle_broadcastable_groups':sum(r['oracle_broadcastable_groups'] for r in rows),
             'mismatches':sum(r['mismatches'] for r in rows),'fraction_checks':sum(r['fraction_checks'] for r in rows),
             'raw_tensor_bytes_total':raw,'summary_payload_bytes_total':sum(r['summary_payload_bytes'] for r in rows),
             'public_results':rows,'planted_control_separate':positive,
             'THEORY_STATUS':'NOT_ESTABLISHED','HARDWARE_STATUS':'NOT_TESTED','CORE_ADMISSION':False,
             'HF_FORWARD':'NOT_TESTED','FULL_KV_RNG':'NOT_TESTED','TARGET_405B':'NOT_TESTED'}
    dump(output/'summary.json',summary)
    dump(output/'weight_manifest.json',{'model':MODEL,'revision':PIN,'tensors':acquisitions})
    if acq.receipts:
        dump(ROOT/'results'/'acquisition_receipts.json',{'receipts':acq.receipts,'total_response_payload_bytes':sum(d['bytes'] for d in acq.receipts),'whole_model_downloaded':False})
    dump(output/'environment_observation.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'runtime_diagnostic_not_latency_benchmark':runtime,'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    manifest={str(p.relative_to(output)).replace('\\','/'):sha(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ['manifest.json','environment_observation.json']}
    dump(output/'manifest.json',manifest)
    print('SUMMARY '+json.dumps({k:v for k,v in summary.items() if k not in ['public_results','planted_control_separate']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results'/'science');p.add_argument('--cache',type=Path,default=ROOT/'.cache');a=p.parse_args();run(a.output,a.cache)
