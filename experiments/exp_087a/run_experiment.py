from __future__ import annotations

import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
from typing import Any

import torch, transformers
from transformers import LlamaForCausalLM

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from vortex_runtime.cross_weight_context_code import aggregate, audit_layer
from vortex_runtime.fixed_public_dynamic_common import DEV_MODEL_ID,DEV_REVISION,PINNED_SAFETENSORS,PINNED_TORCH,PINNED_TRANSFORMERS


def canonical(x: Any)->bytes: return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def digest(b: bytes)->str: return hashlib.sha256(b).hexdigest()
def fsha(p: Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def verify_runtime()->dict[str,str]:
    import safetensors
    actual={"python":platform.python_version(),"torch":torch.__version__,"transformers":transformers.__version__,"safetensors":safetensors.__version__}
    expected={"torch":PINNED_TORCH,"transformers":PINNED_TRANSFORMERS,"safetensors":PINNED_SAFETENSORS}
    bad={k:[expected[k],actual[k]] for k in expected if expected[k]!=actual[k]}
    if bad: raise RuntimeError(f"runtime pin mismatch: {bad}")
    return actual


def load(config:dict[str,Any])->tuple[Any,int]:
    if config['model_id']!=DEV_MODEL_ID or config['revision']!=DEV_REVISION: raise RuntimeError('DEV-W mismatch')
    start=time.perf_counter_ns()
    model=LlamaForCausalLM.from_pretrained(config['model_id'],revision=config['revision'],torch_dtype=torch.bfloat16,attn_implementation=config['attention_implementation'],low_cpu_mem_usage=False).eval()
    resolved=str(getattr(model.config,'_commit_hash','') or '')
    if resolved and resolved!=config['revision']: raise RuntimeError(f"resolved SHA mismatch: {resolved}")
    if model.__class__.__name__!='LlamaForCausalLM': raise RuntimeError('official loader class mismatch')
    return model,time.perf_counter_ns()-start


def execute(config_path:Path,output:Path)->dict[str,Any]:
    config=json.loads(config_path.read_text()); runtime=verify_runtime(); torch.manual_seed(config['seed']);torch.set_num_threads(config['torch_num_threads'])
    model,load_ns=load(config); count=len(model.model.layers)
    if config['layer_selection_rule']!='first_middle_last' or count<3: raise RuntimeError('bad layer rule')
    indices=[0,count//2,count-1]
    start=time.perf_counter_ns(); rows=[]
    for index in indices:
        mlp=model.model.layers[index].mlp
        rows.append(audit_layer(mlp.gate_proj.weight,mlp.up_proj.weight,mlp.down_proj.weight,index,tuple(config['coordinate_periods'])))
    audit_ns=time.perf_counter_ns()-start; summary=aggregate(rows)
    integrity=bool(summary['all_reconstructions_exact'])
    information=(summary['best_information_fraction_p50']<=config['success']['best_information_fraction_p50_max'] and summary['best_information_fraction_p95']<=config['success']['best_information_fraction_p95_max'])
    decision=('INVALID_CROSS_WEIGHT_CONTEXT_CODE_RECONSTRUCTION_FAILURE' if not integrity else 'PROMOTE_CROSS_WEIGHT_CONTEXT_CODE_TO_FUSED_QUERY_GATE' if information else 'REJECT_ALIGNED_CROSS_WEIGHT_CONTEXT_DICTIONARY_AS_COLD_QUERY_CORE')
    core={"schema":config['schema'],"config_sha256":fsha(config_path),"checkpoint":{"model_id":config['model_id'],"revision":config['revision']},"layer_indices":indices,"layer_rows":rows,"aggregate":summary,"gates":{"integrity_passed":integrity,"information_passed":information},"decision":decision}
    result={"experiment":"EXP-087A","name":"cross_weight_context_dictionary_gate","source_sha":os.getenv('GITHUB_SHA','LOCAL_UNVERIFIED'),"evidence_level":"E1","phase":["A-structure","B-reference","C-small-real-checkpoint-weight-observation"],"runtime":runtime,"config_sha256":fsha(config_path),"load_wall_ns":load_ns,"audit_wall_ns":audit_ns,"MEASURED":{"official_checkpoint_loaded":True,"model_forward_calls":0,"selected_layer_indices":indices,"layer_rows":rows},"DERIVED":{"aggregate":summary,"success_thresholds":config['success']},"gates":core['gates'],"authoritative_decision":decision,"deterministic_core_sha256":digest(canonical(core)),"claim_boundary":{"actual_checkpoint_weights":True,"model_forward_calls":0,"ideal_entropy_coder":True,"free_probability_tables_and_decoder":True,"complete_mlp_replaced":False,"dense_arithmetic_reduced":"NOT_TESTED","complete_transformer_layer":"NOT_TESTED","successor_state":"NOT_TESTED","405b_execution":"NOT_TESTED","8gib_gpu":"NOT_TESTED","physical_latency":"NOT_TESTED"},"UNVERIFIED":["cross-layer grammar beyond frozen contexts","sublinear nonlinear checkpoint generator","fused exact query evaluation without materializing weights","attention and LM-head composition","405B same-machine 4B-class p50/p95"]}
    output.mkdir(parents=True,exist_ok=True);(output/'raw').mkdir(exist_ok=True);(output/'artifacts').mkdir(exist_ok=True)
    (output/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    (output/'raw/layer_rows.json').write_text(json.dumps(rows,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    (output/'artifacts/deterministic_core.json').write_bytes(canonical(core))
    paths=['result.json','raw/layer_rows.json','artifacts/deterministic_core.json']
    (output/'checksums.sha256').write_text('\n'.join(f"{fsha(output/p)}  {p}" for p in paths)+'\n')
    return result


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    try: result=execute(args.config,args.output_dir)
    except Exception as exc:
        args.output_dir.mkdir(parents=True,exist_ok=True);result={"experiment":"EXP-087A","source_sha":os.getenv('GITHUB_SHA','LOCAL_UNVERIFIED'),"authoritative_decision":"INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION","error":f"{type(exc).__name__}: {exc}","claim_boundary":{"mechanism_scientifically_rejected":False}};(args.output_dir/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False));return 0
if __name__=='__main__': raise SystemExit(main())
