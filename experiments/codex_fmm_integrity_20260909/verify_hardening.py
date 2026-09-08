from pathlib import Path
import hashlib,json,runpy
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
ns=runpy.run_path(str(P/"verify_repair.py"))
sha=ns["sha"];load=ns["load"];scientific=ns["scientific"]
a=load(P/"red_run/result.json");b=load(P/"corrected_run/result.json");c=load(P/"hardened_run/result.json")
assert scientific(a)==scientific(b)==scientific(c)
assert c["integrity_failures"]==[]
for filename in ("catalog_rows.json","factorization_control_rows.json","selected_plan_rows.json"):
 assert (P/"red_run/raw"/filename).read_bytes()==(P/"hardened_run/raw"/filename).read_bytes()
for line in (P/"hardened_run/checksums.sha256").read_text(encoding="utf-8").splitlines():
 h,rel=line.split("  ",1);assert sha(P/"hardened_run"/rel)==h,rel
for name,h in load(P/"hardened_run/executed_source_manifest.json")["sources"].items():
 snapshot=P/"hardened_sources"/name.replace("/","__")
 assert sha(snapshot)==h,name
 assert (ROOT/name).read_bytes().replace(b"\r\n",b"\n")==snapshot.read_bytes().replace(b"\r\n",b"\n"),name
direct=oracle=0
contract=c["search_contract"]
for row in c["search_rows"]:
 counts=row["state_count_by_depth"];audit=row["accounting"]
 ds=[d for d,n in enumerate(counts,1) if n>0 and d>=contract["minimum_depth"]]
 expected=sum(counts[d-1]*(1+d*len(contract["static_scalar_bytes"])) for d in ds)
 expected_oracle=sum(counts[d-1] for d in ds)
 assert audit["evaluated_direct_plan_count"]==expected
 assert audit["evaluated_oracle_plan_count"]==expected_oracle
 assert audit["completed_evaluated_depths"]==ds
 assert expected==audit["workspace_rejected_count"]+audit["workspace_admissible_count"]
 direct+=expected;oracle+=expected_oracle
out={"current_red_corrected_hardened_old_scientific_fields_equal":True,
 "raw_catalog_control_selected_tables_byte_equal":True,
 "direct_evaluations_expected_and_reported":direct,
 "oracle_evaluations_expected_and_reported":oracle,
 "shape_count":len(c["search_rows"]),"hardened_result_sha256":sha(P/"hardened_run/result.json"),
 "hardened_source_sha256":sha(ROOT/"experiments/exp_100a/run_experiment.py"),
 "focused_tests":load(P/"hardened_focused_validation.json"),
 "scope":"retained/evaluated frozen search only; no native/GPU/global impossibility",
 "THEORY_STATUS":"NOT_ESTABLISHED","HARDWARE_STATUS":"NOT_TESTED","CORE_ADMISSION":False}
(P/"hardening_verification.json").write_bytes((json.dumps(out,indent=2,sort_keys=True)+"\n").encode())
print(json.dumps(out,indent=2))
