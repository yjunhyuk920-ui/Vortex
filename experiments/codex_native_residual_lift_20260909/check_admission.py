from pathlib import Path
import hashlib,json,sys
import numpy as np
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import lift
cases=[
 ("n_zero",np.zeros((1,0),dtype=np.uint8),np.zeros(0,dtype=np.uint8),np.float32(1)),
 ("n_over",np.zeros((1,16385),dtype=np.uint8),np.zeros(16385,dtype=np.uint8),np.float32(1)),
 ("m_zero",np.zeros((0,1),dtype=np.uint8),np.zeros(1,dtype=np.uint8),np.float32(1)),
 ("nonbinary_B",np.array([[2]],dtype=np.uint8),np.array([1],dtype=np.uint8),np.float32(1)),
 ("nonbinary_v",np.array([[1]],dtype=np.uint8),np.array([2],dtype=np.uint8),np.float32(1)),
 ("alpha_low",np.array([[1]],dtype=np.uint8),np.array([1],dtype=np.uint8),lift.f32_from_bits((127-110)<<23)),
 ("alpha_high",np.array([[1]],dtype=np.uint8),np.array([1],dtype=np.uint8),lift.f32_from_bits((127+101)<<23)),
 ("alpha_negative",np.array([[1]],dtype=np.uint8),np.array([1],dtype=np.uint8),np.float32(-1)),
 ("alpha_not_BF16",np.array([[1]],dtype=np.uint8),np.array([1],dtype=np.uint8),np.float32(1+2**-20))]
rows=[]
for name,B,v,a in cases:
 try:lift.run_case(name,B,v,a,a)
 except ValueError as e:rows.append({"name":name,"refused":True,"reason":str(e)})
 else:raise AssertionError(name)
assert lift.round_bf16_ratio(lift.f32_from_bits(0x80000000),np.float32(1))==0
result={"refusals":rows,"negative_zero_decoder":0,"source_sha256":hashlib.sha256((P/"lift.py").read_bytes()).hexdigest(),
 "scope":"domain/refusal boundary only, not graceful allocation failure or a mission fast fallback"}
(P/"admission_validation.json").write_bytes((json.dumps(result,indent=2,sort_keys=True)+"\n").encode())
print("refusals",len(rows),"negative_zero_ok",True)

