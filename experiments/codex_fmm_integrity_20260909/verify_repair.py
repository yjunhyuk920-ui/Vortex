"""Verify saved full runs and independently reconstruct the budget witness."""
from pathlib import Path
from math import prod
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text(encoding="utf-8"))
a=load(P/"red_run/result.json"); b=load(P/"corrected_run/result.json")
assert a["integrity_failures"]==["empty_direct_shape_search"]
assert b["integrity_failures"]==[]
assert b["authoritative_decision"]=="REJECT_CATALOGUED_SMALL_COEFFICIENT_RECTANGULAR_FMM_AS_10X_CORE"
allowed_root={"authoritative_decision","authoritative_decision_scope","integrity_failures","deterministic_core_sha256"}
def scientific(x,path=()):
    if isinstance(x,dict):
        out={}
        for k,v in x.items():
            if not path and k in allowed_root: continue
            if k=="wall_ns" and (path==("environment",) or path[:1]==("search_rows",)): continue
            if k=="accounting" and path[:1]==("search_rows",): continue
            if k in ("direct_selection","oracle_selection") and path[:1] in (
                ("block_rows",),("best_direct_row",),("best_free_transform_oracle_row",)): continue
            out[k]=scientific(v,path+(k,))
        return out
    if isinstance(x,list): return [scientific(v,path+(str(i),)) for i,v in enumerate(x)]
    return x
assert scientific(a)==scientific(b),"undeclared old scientific field changed"
raw_equal={}
for name in ("catalog_rows.json","factorization_control_rows.json","selected_plan_rows.json"):
    assert (P/"red_run/raw"/name).read_bytes()==(P/"corrected_run/raw"/name).read_bytes()
    raw_equal[name]=sha(P/"corrected_run/raw"/name)
snapshots=load(P/"executed_sources.json")
for name,digest in load(P/"corrected_run/executed_source_manifest.json")["sources"].items():
    snapshot=P/snapshots[name]["snapshot"]
    assert sha(snapshot)==digest,name
    # The recorded Windows execution bytes are preserved. Only CRLF transport
    # normalization may distinguish the tracked source on a different checkout.
    if name=="experiments/exp_100a/run_experiment.py":
        # This first corrected source was later hardened; preserve and check
        # its own transport identity rather than equating it to current code.
        assert hashlib.sha256(snapshot.read_bytes().replace(b"\r\n",b"\n")).hexdigest()==load(P/"source_transport.json")["tracked_LF_sha256"]
    else:
        assert (ROOT/name).read_bytes().replace(b"\r\n",b"\n")==snapshot.read_bytes().replace(b"\r\n",b"\n"),name
pre=load(P/"PRE_REPAIR_MANIFEST.json")
for f,k in (("original_runner.py","original_runner_sha256"),("portable_only_runner.py","portable_runner_sha256")):
    assert sha(P/f)==pre[k],f
assert sha(P/snapshots["experiments/exp_100a/config.json"]["snapshot"])==pre["config_sha256"]
assert sha(P/snapshots["vortex_runtime/explicit_rectangular_fmm.py"]["snapshot"])==pre["cost_module_sha256"]
for run in ("red_run","corrected_run"):
    for line in (P/run/"checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest,rel=line.split("  ",1)
        assert sha(P/run/rel)==digest,(run,rel)
empty=[r for r in b["search_rows"] if r["direct_structural_plan_count"]==0]
assert len(empty)==1
cell=empty[0]; audit=cell["accounting"]; witness=audit["minimum_workspace_witness"]
assert audit["evaluated_direct_plan_count"]==audit["workspace_rejected_count"]==2304
assert audit["workspace_admissible_count"]==0
schemes={r["scheme_id"]:r for r in b["scheme_rows"]}
catalog={r["key"]:r for r in load(P/"corrected_run/raw/catalog_rows.json")}
steps=[schemes[i] for i in witness["scheme_ids"]]
m,k,n=cell["block_length"],cell["columns"],cell["rows"]
ap,bp,cp=[prod(s["shape"][i] for s in steps) for i in range(3)]
rp=prod(s["rank"] for s in steps)
lm,lk,ln=[(v+d-1)//d for v,d in zip((m,k,n),(ap,bp,cp))]
cut=witness["cut_depth"]; assert cut==len(steps)==3
# At this full static cut, all remaining recursion products are one.
components={
 "activation_buffer":m*k*2,
 "output_buffer":m*n*2,
 "one_left_form":lm*lk*4,
 "local_weight_workspace":min(lk*ln*2,lk*ln*4),
 "local_output_workspace":min(rp*lm*ln*4,ap*cp*lm*ln*4),
 "factor_metadata":sum(96+3*sum(catalog[s["source_key"]][f]["nnz"] for f in ("u","v","w")) for s in steps),
}
total=sum(components.values())
assert total==audit["minimum_generated_workspace_bytes"]==witness["favorable_workspace_bytes"]
limit=audit["workspace_limit_bytes"]
assert total>limit
out={
 "scientific_old_fields_equal":True,"byte_equal_raw":raw_equal,
 "source_and_run_checksums_verified":True,
 "scope":"finite retained/evaluated EXP100 search only; no full native or physical GPU conclusion",
 "red_result_sha256":sha(P/"red_run/result.json"),
 "corrected_result_sha256":sha(P/"corrected_run/result.json"),
 "shape_count":len(b["search_rows"]),"empty_direct_cell":cell,
 "independent_witness":{"method":"integer component formulas from saved orientation/catalog rows; no call to evaluate_sequence",
 "components":components,"total_bytes":total,"budget_bytes":limit,
 "excess_bytes":total-limit,"total_GiB":total/(2**30),"budget_multiple":total/limit},
 "historical_boundary":"red/corrected current numerical outputs compare exactly; no assertion of byte identity with old hosted run",
 "O1":"OPEN","O2":"OPEN","O3":"OPEN","O4":"OPEN","O5":"OPEN","O6":"PARTIAL",
 "THEORY_STATUS":"NOT_ESTABLISHED","HARDWARE_STATUS":"NOT_TESTED","CORE_ADMISSION":False,
}
(P/"verification.json").write_bytes((json.dumps(out,indent=2,sort_keys=True)+"\n").encode())
print(json.dumps({k:out[k] for k in ("scientific_old_fields_equal","corrected_result_sha256","independent_witness")},indent=2))
