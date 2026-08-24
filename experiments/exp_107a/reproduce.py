from __future__ import annotations

import argparse, hashlib, json, math
from pathlib import Path

OMEGA=2.792481250
TARGET=0.10
BLOCKS=(32,64,128,256,512,1024,2048,4096,8192,16384)
FAMILIES=(
('q',16384,16384,126),('k',16384,1024,126),('v',16384,1024,126),
('o',16384,16384,126),('gate',16384,53248,126),
('up',16384,53248,126),('down',53248,16384,126),
('lm_head',16384,128256,1),
)

def ratio(k:int,omega:float)->float:
    num=den=0.0
    for _,din,dout,count in FAMILIES:
        volume=k*din*dout
        num+=count*volume**(omega/3)
        den+=count*volume
    return num/den

def required_omega(k:int)->float:
    lo,hi=2.0,3.0
    for _ in range(120):
        mid=(lo+hi)/2
        if ratio(k,mid)<=TARGET: lo=mid
        else: hi=mid
    return (lo+hi)/2

def run()->dict:
    rows=[{'block_length':k,'ideal_fraction':ratio(k,OMEGA)} for k in BLOCKS]
    best=min(rows,key=lambda r:r['ideal_fraction'])
    omega_req=required_omega(best['block_length'])
    omega46=3*math.log(46)/math.log(64)
    apa_ideal=ratio(16384,omega46)
    result={
      'schema':'exp-107a-current-fmm-envelope-v1',
      'source_commit':'a326999bae25ae6169aa306c499e61f24acf9a09',
      'catalog':{
        'repository':'dronperminov/FastMatrixMultiplication',
        'commit':'98ba522db92b74f1f8c561a78038ff3091356d73',
        'scheme_count':680,'below_strassen_count':52,
        'best_displayed_normalized_omega':OMEGA,
      },
      'target_families':[{'name':n,'input':i,'output':o,'count':c} for n,i,o,c in FAMILIES],
      'catalog_ideal_envelope':rows,
      'best_catalog_row':best,
      'required_model_weighted_omega':omega_req,
      'omega_gap':OMEGA-omega_req,
      'decision':'REJECT_CURRENT_680_EXPLICIT_FMM_CATALOG_COMPOSITIONS_AS_10X_CORE',
      'apa':{
        'format':'4x4x4','border_rank':46,'reported_polynomial_order':3,
        'normalized_omega':omega46,
        'zero_overhead_model_fraction':apa_ideal,
        'maximum_exactification_multiplier':TARGET/apa_ideal,
        'registered_tensor_power':7,
        'registered_generic_degree':21,
        'registered_generic_evaluations':22,
        'generic_interpolation_model_indicator':22*apa_ideal,
        'decision':'REJECT_REGISTERED_NAIVE_RANK46_APA_INTERPOLATION_EXACTIFICATION',
        'scope':'not a lower bound on every border-rank exactification',
      },
      'lupanov_indicator':{
        'q4_independent_bitplanes_over_log2_16384':4/14,
        'bf16_independent_bitplanes_over_log2_16384':16/14,
        'scope':'optimistic construction indicator, not a circuit lower bound',
      },
      'next_survivor':'FINITE_STRASSEN_CALCULUS_DIRECT_SUM_EXTRACTION_CHEAP_KILL_ONLY',
      'integrity_failures':[],
      'claim_boundary':{
        '405b':'NOT_TESTED','8gib':'NOT_TESTED','causal_A_N_over_A':'NOT_TESTED',
        'cuda':'NOT_TESTED','final_latency':'NOT_TESTED'
      },
    }
    if best['block_length']!=16384 or best['ideal_fraction']<=TARGET: result['integrity_failures'].append('catalog_gate')
    if OMEGA<=omega_req: result['integrity_failures'].append('omega_order')
    if apa_ideal>=TARGET or 22*apa_ideal<=TARGET: result['integrity_failures'].append('apa_gate')
    core=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)
    result['deterministic_core']=hashlib.sha256(core.encode()).hexdigest()
    return result

def self_test()->None:
    a,b=run(),run()
    assert a==b and not a['integrity_failures']
    assert abs(a['best_catalog_row']['ideal_fraction']-0.125150870059609)<1e-15
    assert abs(a['required_model_weighted_omega']-2.7700683089152998)<1e-12
    assert a['apa']['zero_overhead_model_fraction']<0.1<a['apa']['generic_interpolation_model_indicator']
    assert a['lupanov_indicator']['q4_independent_bitplanes_over_log2_16384']>0.1

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--self-test',action='store_true'); args=p.parse_args()
    if args.self_test: self_test()
    out=run(); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'decision':out['decision'],'best_fraction':out['best_catalog_row']['ideal_fraction'],'required_omega':out['required_model_weighted_omega'],'next':out['next_survivor'],'core':out['deterministic_core']},indent=2))
