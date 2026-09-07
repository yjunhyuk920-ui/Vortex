from __future__ import annotations
import ctypes,hashlib,itertools,json,math,platform,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from residue import Program,compile_matrix,bf16_bits_int,generic_scale

def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank_mod(w,p=65521):
 a=np.asarray(w,dtype=np.int64).copy()%p;m,n=a.shape;r=0
 for c in range(n):
  pos=next((k for k in range(r,m) if a[k,c]),None)
  if pos is None:continue
  a[[r,pos]]=a[[pos,r]]
  a[r]=a[r]*pow(int(a[r,c]),-1,p)%p
  factors=a[r+1:,c].copy();a[r+1:]=(a[r+1:]-factors[:,None]*a[r])%p
  r+=1
  if r==m:break
 return r

def main():
 out=ROOT/'results'/'run';out.mkdir(parents=True,exist_ok=True)
 libpath=ROOT/'results'/'reference.so'
 subprocess.run(['cc','-O2','-std=c11','-ffp-contract=off','-fno-fast-math','-shared','-fPIC',str(ROOT/'src/reference.c'),'-o',str(libpath)],check=True)
 lib=ctypes.CDLL(str(libpath));lib.ref.argtypes=[ctypes.POINTER(ctypes.c_int16),ctypes.POINTER(ctypes.c_int8),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_uint16)]
 configs=[];total=values=mismatches=all_dense=0
 for n in [16,64,128]:
  for seed in [1701,1702]:
   for group in ['dense_uniform','dense_small_residual','constant_rows']:
    rng=np.random.default_rng(seed+100*n)
    if group=='dense_uniform':w=rng.integers(-63,64,size=(n,n),dtype=np.int16)
    elif group=='dense_small_residual':
     base=rng.integers(2,33,size=(n,1),dtype=np.int16)
     w=base+rng.integers(-1,2,size=(n,n),dtype=np.int16)
    else:w=np.repeat(rng.integers(1,33,size=(n,1),dtype=np.int16),n,axis=1)
    xs=np.asarray([[0]*n,[1]*n,[-1]*n,[1 if j%2 else -1 for j in range(n)]]+rng.integers(-1,2,size=(12,n)).tolist(),dtype=np.int8)
    d=out/f'{group}_{n}_{seed}';d.mkdir(exist_ok=True)
    np.save(d/'weights.npy',w);np.save(d/'inputs.npy',xs)
    native_w_bits=(w.astype(np.float32).view(np.uint32)>>16).astype('<u2')
    (d/'weights.bf16').write_bytes(native_w_bits.tobytes())
    blob,meta=compile_matrix(w.tolist());(d/'program.bin').write_bytes(blob)
    program=Program((d/'program.bin').read_bytes())
    records=[];ref=np.empty(n,dtype=np.uint16)
    for qi,x in enumerate(xs):
     y,cost=program.run(x.tolist());exact=(w.astype(np.int64)@x.astype(np.int64)).tolist()
     lib.ref(w.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),x.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),n,n,ref.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
     bits=[bf16_bits_int(z) for z in y]
     mm=sum(a!=b for a,b in zip(y,exact))+sum(a!=int(b) for a,b in zip(bits,ref))
     mismatches+=mm;total+=1;values+=n
     support=sum(ev['lifted_residual']!=0 for ev in cost['events']);all_dense+=support==n
     records.append({'query':qi,'exact_output':exact,'returned_output':y,'source_bf16':bits,'c_bf16':ref.tolist(),'mismatches':mm,'correction_support':support,'cost':cost})
    save(d/'records.json',records);save(d/'metadata.json',meta)
    info={'group':group,'n':n,'seed':seed,'rank_lower_bound_mod65521':rank_mod(w),
          'queries':len(xs),'coordinates':len(xs)*n,'mismatches':sum(t['mismatches'] for t in records),
          'source_bytes':len(blob),'original_bf16_bytes':2*n*n,
          'source_read_ratio_bf16':len(blob)/(2*n*n),
          'bitpacked_original_payload':meta['signed_bitpacked_original_payload'],
          'source_ratio_bitpacked_original_payload':len(blob)/meta['signed_bitpacked_original_payload'],
          'residue_bits_range':[min(r['modulus_bits'] for r in meta['rows']),max(r['modulus_bits'] for r in meta['rows'])],
          'stored_bits_range':[min(r['stored_signed_bits'] for r in meta['rows']),max(r['stored_signed_bits'] for r in meta['rows'])],
          'all_coordinates_corrected_queries':sum(t['correction_support']==n for t in records),
          'coefficient_unique_lift':all(r['residue_coefficient_lift_recovers_every_residual'] for r in meta['rows']),
          'first8_min_source_bytes_ratio':(2*n*n+len(blob)+8*len(blob))/(8*2*n*n),
          'weights_sha256':h(d/'weights.npy'),'inputs_sha256':h(d/'inputs.npy'),'program_sha256':h(d/'program.bin')}
    configs.append(info)
 save(out/'summary.json',{'matrices':len(configs),'queries':total,'coordinates':values,
      'mismatches_integer_and_bf16':mismatches,'all_coordinates_corrected_queries':all_dense,
      'configs':configs,'THEORY_STATUS':'NOT_ESTABLISHED','CORE_ADMISSION':False,'HARDWARE_STATUS':'NOT_TESTED',
      'scope':'registered integer BF16 / ternary inputs; each row positive coefficient. Not arbitrary floats, HF, KV/RNG or latency.'})
 # Scope B: congruence is not closed through a wide nonlinear map. ReLU is exact.
 scope={'nonlinear_congruence':{'q':4,'state_a':-1,'state_b':3,'same_input_residue':3,
           'relu_a':0,'relu_b':3,'relu_output_residues':[0,3],
           'conclusion':'No single residue-state update exists for ReLU on this domain; not a theorem about all encodings.'},
        'native_sum_warning':{'products_a':[2**24,1,-2**24,1],'products_b':[2**24,-2**24,1,1],
           'exact_sum_both':2,'reference_a':float(np.float32(np.float32(2**24)+np.float32(1))+np.float32(np.float32(-2**24)+np.float32(1))),
           'reference_b':float(np.float32(np.float32(2**24)+np.float32(-2**24))+np.float32(np.float32(1)+np.float32(1))),
           'scope':'Outside the admitted exact integer-accumulation range; exact total and all its residues do not reproduce arbitrary native trees.'}}
 # Scope C: codewords only, fixed side info. No claim when decoder can evaluate W,x.
 H=np.random.default_rng(610).integers(0,2,size=(4,8),dtype=np.int64)
 seen={};first=None;count=0
 for e in itertools.product((-1,0,1),repeat=8):
  syn=tuple((H@np.array(e,dtype=np.int64)%2).tolist());count+=1
  if syn in seen and first is None:first={'error_a':seen[syn],'error_b':list(e),'syndrome':list(syn)}
  seen.setdefault(syn,list(e))
 scope['joint_decoder']={'H':H.tolist(),'errors':count,'distinct_syndromes':len(seen),'first_collision':first,
      'necessary_total_bits':math.ceil(8*math.log2(3)),
      'assumption':'decoder receives only fixed coarse side info and syndrome; arbitrary error cube; x-dependent decoding can invalidate this restricted counting model'}
 # Native SiLU illustration only; not used by the exact source or its proof.
 def bf16_fp32(v):
  u=int(np.asarray(np.float32(v)).view(np.uint32));return ((u+0x7fff+((u>>16)&1))>>16)&65535
 scope['silu_sign_witness']={'inputs':[-1,3],'modulus':4,'equal_input_residue':3,
     'bf16_bits_after_fp32_eval':[bf16_fp32(z/(1+math.exp(-z))) for z in (-1,3)],
     'note':'Supplementary illustrative scalar values from math.exp then FP32/BF16; no CUDA or all-input exp proof. SiLU has opposite signs here.'}
 save(out/'scope.json',scope)
 save(out/'scale.json',{'cases':[generic_scale(n) for n in [16,64,128,1024,16384]],
      'coefficient_recovery_theorem':'If q>2 ||R_i||_1, then each R_ij is the unique representative of R_ij mod q in [-||R_i||_1,||R_i||_1]. Full coefficient information remains; not a cell-probe lower bound.',
      'global_target':'O1-O6 open; comparison of fields/ops only; no target or ordinary native float execution claim'})
 manifest={str(p.relative_to(out)):h(p) for p in sorted(out.rglob('*')) if p.is_file()}
 save(ROOT/'results'/'manifest.json',manifest)
 print(json.dumps({'queries':total,'coordinates':values,'mismatches':mismatches,'files':len(manifest),'all_dense_queries':all_dense}))
 if mismatches:raise SystemExit('mismatch')
if __name__=='__main__':main()
