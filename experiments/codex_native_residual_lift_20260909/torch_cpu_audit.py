"""Small actual CPU operator audit; no full HF/causal-state/target test."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,platform,sys,time
import numpy as np
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
def exact_rne_ratio(y,a):return round(Fraction.from_float(float(y))/Fraction.from_float(float(a)))
def sha_tensor(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
rows=[]
for n,counts in ((257,[256,257]),(16384,[16320,16384,8193,0])):
 b=n.bit_length();m=len(counts);D=n+m*b
 array=np.zeros((m,D),dtype=np.float32)
 for i,k in enumerate(counts):
  array[i,:k]=1
  for j in range(b):array[i,n+i*b+j]=-(1<<j)
 C=torch.from_numpy(array).to(torch.bfloat16)
 source_sha=sha_tensor(C)
 for a0,a1 in ((1.0,1.0),(1.0078125,1.9921875),(2.0**-100,1.9921875*2.0**100)):
  for op_name in ('linear','mv'):
   x0=torch.zeros(D,dtype=torch.bfloat16);x0[:n]=a0
   state=torch.get_rng_state().clone()
   begin=time.perf_counter_ns()
   def query(x):
    return F.linear(x.unsqueeze(0),C).squeeze(0) if op_name=='linear' else torch.mv(C,x)
   y0=query(x0)
   k0=[min(n,max(0,exact_rne_ratio(y,a0))) for y in y0]
   x1=torch.zeros(D,dtype=torch.bfloat16);x1[:n]=a1
   for i,k in enumerate(k0):
    for j in range(b):
     if (k>>j)&1:x1[n+i*b+j]=a1
   y1=query(x1)
   decoded=[k+exact_rne_ratio(y,a1) for k,y in zip(k0,y1)]
   elapsed=time.perf_counter_ns()-begin
   assert torch.equal(state,torch.get_rng_state())
   assert sha_tensor(C)==source_sha
   expected_y0=torch.tensor([a0*k for k in counts],dtype=torch.bfloat16)
   expected_y1=torch.tensor([a1*(k-s) for k,s in zip(counts,k0)],dtype=torch.bfloat16)
   words=lambda t:[f'{int(x)&65535:04x}' for x in t.view(torch.int16).tolist()]
   rows.append({"n":n,"m":m,"D":D,"operator":op_name,"alpha0":a0,"alpha1":a1,
    "counts":counts,"k0":k0,"decoded":decoded,"exact_decode":decoded==counts,
    "y0_words":words(y0),"y1_words":words(y1),"expected_y0_words":words(expected_y0),
    "expected_y1_words":words(expected_y1),"matches_declared_FP32_sum_then_BF16_store":
       torch.equal(y0,expected_y0) and torch.equal(y1,expected_y1),
    "rng_preserved":True,"coefficient_sha256_before_and_after":source_sha,
    "coefficient_payload_bytes":C.numel()*C.element_size(),
    "two_full_matrix_reads_logical_bytes":2*C.numel()*C.element_size(),
    "elapsed_control_ns":elapsed,"timing_scope":"CPU harness incl input/decode; not a latency benchmark"})
out={"torch":torch.__version__,"numpy":np.__version__,"python":sys.version,"platform":platform.platform(),
 "device":"cpu","cuda_executed":False,"threads":torch.get_num_threads(),
 "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"rows":rows,
 "native_call_count":2*len(rows),"operator_case_count":len(rows),
 "mismatch_cases":[{k:r[k] for k in ("operator","n","alpha0","alpha1","counts","decoded")} for r in rows if not r["exact_decode"]],
 "scope":"actual small CPU operators only; not a full HF/RNG-generation/KV/causal-token/all-continuation lift",
 "ABI_rule":"Agreement checks this CPU configuration; a mismatching operator is outside the declared FP32-intermediate ABI, not silently repaired."}
target=P/"torch_cpu_results.json"
if target.exists():raise FileExistsError(target)
target.write_bytes((json.dumps(out,indent=2,sort_keys=True)+"\n").encode())
print(json.dumps({k:out[k] for k in ("torch","native_call_count","operator_case_count","mismatch_cases")},indent=2))

