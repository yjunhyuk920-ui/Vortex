"""Primary integer audit; this does not execute a floating-point matvec."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
def rne_div(n,d):
 q,r=divmod(abs(n),d)
 q+=2*r>d or (2*r==d and q%2==1)
 return -q if n<0 else q
def bf16_significand_round(t):
 if not t:return 0
 shift=max(0,abs(t).bit_length()-8)
 return rne_div(t,1<<shift)*(1<<shift)
maximum=0;witness=None;first_cases=0
for mantissa in range(128,256):
 for count in range(16385):
  y=bf16_significand_round(mantissa*count)
  k0=min(16384,max(0,rne_div(y,mantissa)))
  delta=count-k0;first_cases+=1
  if abs(delta)>maximum:maximum=abs(delta);witness={"mantissa":mantissa,"count":count,"rounded_scaled_integer":y,"k0":k0,"delta":delta}
  assert abs(delta)<=65
second_cases=0
for mantissa in range(128,256):
 for delta in range(-65,66):
  y=bf16_significand_round(mantissa*delta)
  assert rne_div(y,mantissa)==delta
  second_cases+=1
# Pure numerical substitution into registered target projection dimensions.
root=P.parents[1]
families=json.loads((root/"experiments/codex_fmm_integrity_20260909/hardened_run/result.json").read_text())["target_families"]
scale=[]
for family in families:
 rows,cols,count=family["rows"],family["columns"],family["count"]
 choices=[(n*min(rows,(cols-n)//n.bit_length()),n,min(rows,(cols-n)//n.bit_length())) for n in range(1,min(16384,cols-1)+1)]
 freedom,n,m=max(choices)
 b=n.bit_length()
 scale.append({"family":family["name"],"rows":rows,"columns":cols,"count":count,"binary_source_columns":n,
 "source_rows":m,"bias_bits_per_row":b,"active_input_width":n+m*b,"independent_source_bits_per_matrix":freedom,
 "registered_parameters":rows*cols*count,"independent_source_bits_population":freedom*count,
 "logical_full_registered_BF16_bytes":2*rows*cols*count,"two_native_full_matrix_reads_bytes":4*rows*cols*count,
 "source_fraction_of_registered_weight_slots":freedom/(rows*cols),
 "scope":"packing of a structured reduction family only, not arbitrary checkpoint compression or a global advice allocation"})
out={"method":"finite integer audit after proving exponent-normal homogeneity; not a native kernel measurement",
 "first_cases":first_cases,"second_cases":second_cases,"maximum_observed_first_residual":maximum,"maximum_witness":witness,
 "scale_rows":scale,"total_registered_parameters":sum(x["registered_parameters"] for x in scale),
 "total_embedded_source_bits":sum(x["independent_source_bits_population"] for x in scale),
 "two_native_full_registered_matrix_reads_bytes":sum(x["two_native_full_matrix_reads_bytes"] for x in scale),
 "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 "scope":"auxiliary O3 reduction; two dense queries, no subdense constructor/lower bound/latency proof"}
target=P/"integer_audit.json"
if target.exists():raise FileExistsError(target)
target.write_bytes((json.dumps(out,indent=2,sort_keys=True)+"\n").encode())
print(json.dumps({k:out[k] for k in ("first_cases","second_cases","maximum_observed_first_residual","maximum_witness","total_registered_parameters","total_embedded_source_bits","two_native_full_registered_matrix_reads_bytes")},indent=2))

