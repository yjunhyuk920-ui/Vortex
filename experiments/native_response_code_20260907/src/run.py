from __future__ import annotations
import argparse, ctypes, hashlib, json, platform, subprocess
from pathlib import Path
import numpy as np
import torch
from native import bf_bits,value,model,scalar_model,alphabet,inputs
from response import encode,Program,mobius,pulse_query,runs,run_query
ROOT=Path(__file__).resolve().parents[1]

def savejson(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def perc(a,p):return float(np.percentile(a,p,method='higher'))

def load_c(out):
    lib=out/'eval.so'
    subprocess.run(['gcc','-O2','-std=c11','-Wall','-Wextra','-Werror','-shared','-fPIC',str(ROOT/'src/eval.c'),'-o',str(lib)],check=True)
    f=ctypes.CDLL(str(lib)).vx_query
    f.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t,ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_uint16)]
    f.restype=ctypes.c_int
    return f

def exercise(name,table,alpha,xbits,out,cf,weight_bytes=0,mac=0,modelinfo=None):
    folder=out/name;folder.mkdir()
    table=np.ascontiguousarray(table,dtype='<u2');np.save(folder/'response.npy',table)
    blob,meta=encode(table,alpha);(folder/'program.bin').write_bytes(blob)
    p=Program.load(blob)
    cb=(ctypes.c_uint8*len(blob)).from_buffer_copy(blob)
    yn=np.empty(p.m,np.uint16); yptr=yn.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    trace=[]; cgood=pgood=0
    for idx,xx in enumerate(xbits):
        yy,cost=p.query(xx)
        assert np.array_equal(yy,table[idx]),(name,idx,'Python mismatch')
        rc=cf(cb,len(blob),xx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),yptr)
        assert rc==0 and np.array_equal(yn,table[idx]),(name,idx,'C mismatch',rc)
        trace.append((cost['atoms'],cost['payload_bytes'],cost['query_reference_bytes'],cost['domain_comparisons']))
        pgood+=1;cgood+=1
    tr=np.asarray(trace,dtype=np.uint32);np.save(folder/'query_costs.npy',tr)
    np.save(folder/'left_codes.npy',np.asarray([str(c) for c in meta.pop('codes')]))
    savejson(folder/'basis_certificate.json',meta)
    coeff=mobius(table);assert np.array_equal(mobius(coeff),table)
    np.save(folder/'pulse_coefficients.npy',coeff)
    active=int(np.count_nonzero(np.any(coeff!=0,axis=1)))
    selected=np.unique(np.linspace(0,len(xbits)-1,min(64,len(xbits)),dtype=int))
    pulse_reads=[]
    for idx in selected:
        yy,reads=pulse_query(coeff,int(idx));assert np.array_equal(yy,table[idx]);pulse_reads.append(reads)
    starts,vals=runs(table);np.save(folder/'run_starts.npy',starts);np.save(folder/'run_values.npy',vals)
    recon=vals[np.searchsorted(starts,np.arange(len(table)),side='right')-1]
    assert np.array_equal(recon,table)
    for idx in selected:
        yy,_=run_query(starts,vals,int(idx));assert np.array_equal(yy,table[idx])
    QN=len(table); L=p.L; R=p.R; fullbits=int(np.log2(QN))
    metrics={
      'name':name,'queries':QN,'output_coordinates':int(table.size),'python_mismatches':0,'c_mismatches':0,
      'rank':p.r,'maximum_row_rank':L,'raw_table_bytes':int(table.nbytes),'program_file_bytes':len(blob),
      'weight_bytes':weight_bytes,'baseline_MAC_per_input':mac,'native_source_evaluations':QN if weight_bytes else 0,
      'source_MAC':mac*QN,'source_raw_output_write_bytes':int(table.nbytes),
      'source_transient_note':'Full inputs/response, factor elimination big integers, LUT and weights coexist; no <=8GiB claim.',
      'query_payload_bytes_p50':perc(tr[:,1],50),'query_payload_bytes_p95':perc(tr[:,1],95),
      'query_references_bytes_p50':perc(tr[:,2],50),'query_references_bytes_p95':perc(tr[:,2],95),
      'atoms_p50':perc(tr[:,0],50),'atoms_p95':perc(tr[:,0],95),'atoms_max':int(tr[:,0].max()),
      'pulse':{'nonzero_word_coefficients':active,'all_coefficients':QN,'dense_payload_bytes':int(coeff.nbytes),
               'explicit_query_tests':len(selected),'reconstruction_all_coordinates':int(table.size),
               'mean_payload_bytes_all_addresses':float(2*p.m*(1.5**fullbits)),
               'sparse_mask_payload_bytes':active*(4+2*p.m)},
      'runs':{'constant_runs':len(starts),'payload_bytes':int(starts.nbytes+vals.nbytes),'all_queries_match':True},
      'original_weights_in_program':False,'activation_LUT_in_program':False,
      'model':modelinfo,'core_admission':False}
    if weight_bytes:
        metrics.update({'program_to_weight_ratio':len(blob)/weight_bytes,
                        'query_reference_p50_to_weight_ratio':metrics['query_references_bytes_p50']/weight_bytes,
                        'query_reference_p95_to_weight_ratio':metrics['query_references_bytes_p95']/weight_bytes,
                        # Output writer/evaluator has weights resident. This is arithmetic cost,
                        # NOT a claim that the constructor reads W QN times from SSD.
                        'construction_MAC_over_single_original_call':QN,
                        'cold_program_validation_read_bytes':len(blob),
                        'cold_min_weight_read_plus_program_write_bytes':weight_bytes+len(blob)})
    savejson(folder/'metrics.json',metrics)
    print(name,'rank',p.r,'code/weight',metrics.get('program_to_weight_ratio'),'query p50',metrics['query_references_bytes_p50'],flush=True)
    return metrics

def run(out):
    if out.exists() and any(out.iterdir()):raise ValueError('output directory must be empty')
    out.mkdir(parents=True,exist_ok=True);cf=load_c(out)
    torch.set_num_threads(1)
    allb=np.arange(65536,dtype=np.uint16)
    with torch.no_grad():
        tv=torch.from_numpy(allb.view(np.int16).copy()).view(torch.bfloat16)
        lut=torch.nn.functional.silu(tv).view(torch.int16).numpy().view(np.uint16).copy()
    np.save(out/'silu_abi.npy',lut)
    summaries=[];seen=set();scalar_count=0
    fixtures=[(q,32,seed) for q in (2,4,8,16) for seed in (7,19)] +[(8,256,s) for s in (7,19)]
    for q,h,seed in fixtures:
        name=f'q{q}_h{h}_s{seed}';alpha=alphabet(q);xs=inputs(alpha)
        rng=np.random.default_rng(seed)
        def coeff(shape):
            vals=rng.integers(1,8,size=shape)*rng.choice([-1,1],size=shape)
            return bf_bits(vals/8.)
        G,U,D=coeff((h,4)),coeff((h,4)),coeff((4,h))
        tables=[]
        for off in range(0,len(xs),256):
            y,g=model(xs[off:off+256],G,U,D,lut);tables.append(y);seen.update(map(int,np.unique(g)))
        table=np.concatenate(tables)
        ids=np.unique(np.linspace(0,len(xs)-1,min(64,len(xs)),dtype=int))
        for idx in ids:
            independent=np.asarray(scalar_model(xs[idx],G,U,D,lut),dtype=np.uint16)
            assert np.array_equal(independent,table[idx]),('native scalar mismatch',name,int(idx));scalar_count+=1
        weight_bytes=G.nbytes+U.nbytes+D.nbytes;mac=G.size+U.size+D.size
        info={'Q':q,'n':4,'m':4,'h':h,'seed':seed,'independent_scalar_inputs':len(ids),'unmodified_fixture':True}
        metrics=exercise(name,table,alpha,xs,out,cf,weight_bytes,mac,info)
        folder=out/name
        for nm,a in [('G',G),('U',U),('D',D),('alphabet',alpha),('inputs',xs)]:np.save(folder/f'{nm}.npy',a)
        summaries.append(metrics)
    # Mechanism controls are not neural network acceleration results.
    alpha=alphabet(4);xs=inputs(alpha);N=len(xs)
    tables={'identity_words':xs.copy(),'signed_zero_words':np.where((np.arange(N)[:,None]>>(np.arange(4)))&1,0x8000,0).astype(np.uint16),
            'zero_words':np.zeros((N,4),np.uint16)}
    for name,table in tables.items(): summaries.append(exercise(name,table,alpha,xs,out,cf,modelinfo={'control_only':True}))
    single_mismatch=[]
    with torch.no_grad():
        for bit in sorted(seen):
            a=np.array([bit],np.uint16)
            b=torch.nn.functional.silu(torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16)).view(torch.int16).numpy().view(np.uint16)[0]
            if int(b)!=int(lut[bit]):single_mismatch.append(bit)
    assert not single_mismatch,'activation single/batch mismatch: ABI must not silently change'
    totalq=sum(m['queries'] for m in summaries if m['weight_bytes'])
    summary={'abi':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'CPU',
        'silu_primitive':'torch CPU bfloat16 silu frozen one-dimensional table; not mathematical correctly-rounded SiLU proof',
        'silu_table_sha256':sha(out/'silu_abi.npy'),'silu_encountered_singleton_checks':len(seen),'silu_singleton_mismatches':0},
        'main_models':len(fixtures),'main_queries':totalq,'main_output_coordinates':totalq*4,
        'independent_scalar_inputs':scalar_count,'full_domain_python_mismatches':0,'full_domain_c_mismatches':0,
        'all_sources_generated_explicitly':True,'full_bf16_input_domain_supported':False,'fixtures':summaries,
        'THEORY_STATUS':'NOT_ESTABLISHED','HARDWARE_STATUS':'NOT_TESTED','CORE_ADMISSION':False,
        'THREE_QUALIFYING_NEW_PRINCIPLES':False,'FULL_MISSION_O1_O6':'OPEN'}
    savejson(out/'summary.json',summary)
    # A bound on the exhaustive constructor, not on all representations/programs.
    savejson(out/'scale.json',{'source_evaluations':'Q**n','two_half_dimension':'Q**(n/2)',
         'n16384_binary_source_evaluations_log2':16384,'n16384_binary_half_dimension_log2':8192,
         'n16384_all16bit_source_evaluations_log2':16*16384,
         'interpretation':'This explicit constructor is exponential even with two allowed values per coordinate. No large compile attempted.',
         'memory_8GiB_closed':False,'latency4B_closed':False,'TTFT_closed':False})
    print('MAIN:',totalq,'queries',totalq*4,'outputs;',scalar_count,'scalar references; singleton SiLU',len(seen),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/run');run(p.parse_args().out.resolve())
