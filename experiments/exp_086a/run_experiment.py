from __future__ import annotations

import argparse, hashlib, json, math, os, platform, sys, time
from pathlib import Path
from typing import Any

import torch, transformers
from transformers import AutoTokenizer, LlamaForCausalLM

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from vortex_runtime.fixed_public_dynamic_common import DEV_MODEL_ID,DEV_REVISION,PINNED_SAFETENSORS,PINNED_TORCH,PINNED_TRANSFORMERS
from vortex_runtime.functional_microprogram_compiler import P50,fit_library,residual_diagnostics,target_resources


def sha(data:bytes)->str: return hashlib.sha256(data).hexdigest()
def fsha(p:Path)->str: return sha(p.read_bytes())
def tsha(t:torch.Tensor)->str: return sha(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
def canon(x:Any)->bytes: return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def runtime()->dict[str,str]:
 import safetensors
 a={'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'safetensors':safetensors.__version__}
 e={'torch':PINNED_TORCH,'transformers':PINNED_TRANSFORMERS,'safetensors':PINNED_SAFETENSORS}
 bad={k:[e[k],a[k]] for k in e if e[k]!=a[k]}
 if bad: raise RuntimeError(f'pin mismatch: {bad}')
 return a


def load(cfg):
 if cfg['model_id']!=DEV_MODEL_ID or cfg['revision']!=DEV_REVISION: raise RuntimeError('DEV-W mismatch')
 t=time.perf_counter_ns()
 m=LlamaForCausalLM.from_pretrained(cfg['model_id'],revision=cfg['revision'],torch_dtype=torch.bfloat16,attn_implementation='eager',low_cpu_mem_usage=False).eval()
 tok=AutoTokenizer.from_pretrained(cfg['model_id'],revision=cfg['revision'],use_fast=True)
 return m,tok,time.perf_counter_ns()-t


def capture(model,tok,prompt,layer,positions):
 xs=[]; ys=[]
 def pre(_m,args): xs.append(args[0][:,-1,:].detach().cpu().contiguous())
 def post(_m,_a,out): ys.append(out[:,-1,:].detach().cpu().contiguous())
 mlp=model.model.layers[layer].mlp; h1=mlp.register_forward_pre_hook(pre); h2=mlp.register_forward_hook(post); tokens=[]
 try:
  ids=tok(prompt,return_tensors='pt')['input_ids']
  with torch.inference_mode():
   out=model(input_ids=ids,use_cache=True,return_dict=True); past=out.past_key_values; nxt=int(out.logits[0,-1].argmax()); tokens.append(nxt)
   for _ in range(1,positions):
    out=model(input_ids=torch.tensor([[nxt]]),past_key_values=past,use_cache=True,return_dict=True); past=out.past_key_values; nxt=int(out.logits[0,-1].argmax()); tokens.append(nxt)
 finally: h1.remove(); h2.remove()
 if len(xs)!=positions or len(ys)!=positions: raise RuntimeError('capture count mismatch')
 return torch.cat(xs).bfloat16(),torch.cat(ys).bfloat16(),tokens


def qjson(q):
 return {'mode':q['mode'],'precision':q['precision'],'selected':q['selected'],'use_counts':q['use_counts'],'report':q['report'],'candidate_sha256':tsha(q['candidate'])}


def execute(config_path:Path,outdir:Path):
 cfg=json.loads(config_path.read_text()); rt=runtime(); torch.manual_seed(cfg['seed']); torch.set_num_threads(cfg['torch_num_threads'])
 model,tok,load_ns=load(cfg); build_x=[];build_y=[];eval_x=[];eval_y=[];traces=[]; t=time.perf_counter_ns()
 for s in cfg['prompts']:
  x,y,tokens=capture(model,tok,s['prompt'],cfg['layer_index'],cfg['decode_positions_per_prompt'])
  (build_x if s['split']=='build' else eval_x).append(x); (build_y if s['split']=='build' else eval_y).append(y)
  traces.append({'id':s['id'],'family':s['family'],'split':s['split'],'prompt_sha256':sha(s['prompt'].encode()),'states':x.shape[0],'input_sha256':tsha(x),'output_sha256':tsha(y),'tokens':tokens})
 capture_ns=time.perf_counter_ns()-t; xb,yb,xe,ye=map(torch.cat,(build_x,build_y,eval_x,eval_y))
 t=time.perf_counter_ns(); lib,assign=fit_library(xb,yb,cfg['clusters'],cfg['rank_cap']); compile_ns=time.perf_counter_ns()-t
 rawq={
  'build_oracle_fp64':lib.query(xb,yb,'fp64',True),'build_router_fp64':lib.query(xb,yb,'fp64',False),'build_router_fp32':lib.query(xb,yb,'fp32',False),
  'eval_oracle_fp64':lib.query(xe,ye,'fp64',True),'eval_oracle_fp32':lib.query(xe,ye,'fp32',True),'eval_router_fp32':lib.query(xe,ye,'fp32',False)}
 queries={k:qjson(v) for k,v in rawq.items()}; res=target_resources(cfg['clusters'],cfg['rank_cap'],cfg['sidecar_scalar_bytes'])
 build_counts=[int(v) for v in torch.bincount(assign,minlength=cfg['clusters']).tolist()]
 used=[v for v in rawq['eval_oracle_fp32']['use_counts'] if v>0]
 integrity=queries['build_oracle_fp64']['report']['vector_exact_fraction']==1 and queries['build_router_fp64']['report']['vector_exact_fraction']==1
 resource=res['sidecar_bytes']<=cfg['target_sidecar_limit_bytes'] and res['compiled_mlp_operation_fraction']<=cfg['success']['compiled_mlp_operation_fraction_max']
 oracle=queries['eval_oracle_fp32']['report']['vector_exact_fraction']==cfg['success']['oracle_vector_exact_fraction_min'] and bool(used) and min(used)>=cfg['success']['minimum_evaluation_states_per_used_program']
 router=queries['eval_router_fp32']['report']['vector_exact_fraction']==cfg['success']['router_vector_exact_fraction_min']
 if not integrity: decision='INVALID_CAUSAL_FUNCTIONAL_MICROPROGRAM_NUMERICAL_CONTROL_FAILURE'
 elif not resource: decision='REJECT_CAUSAL_FUNCTIONAL_MICROPROGRAM_RESOURCE_GATE'
 elif not oracle: decision='REJECT_LOW_RANK_FUNCTIONAL_MICROPROGRAM_LIBRARY_AT_ORACLE_GATE'
 elif not router: decision='PROMOTE_FUNCTIONAL_MICROPROGRAM_LIBRARY_TO_CAUSAL_ROUTER_GATE'
 else: decision='PROMOTE_FUNCTIONAL_MICROPROGRAM_TO_COMPLETE_MLP_REPLACEMENT_GATE'
 manifest={'mechanism':'checkpoint_static_low_rank_affine_whole_swiglu_microprogram_library','clusters':cfg['clusters'],'rank_cap':cfg['rank_cap'],'fingerprint':lib.fingerprint,'programs':[{'rank':p.rank,'build_count':p.build_count,'fingerprint':p.fingerprint} for p in lib.programs]}
 core={'config_sha256':fsha(config_path),'manifest':manifest,'traces':traces,'queries':queries,'resources':res,'gates':{'integrity':integrity,'resource':resource,'oracle':oracle,'router':router},'decision':decision}
 result={'experiment':'EXP-086A','name':'causal_functional_microprogram_sharing_gate','source_sha':os.getenv('GITHUB_SHA','LOCAL_UNVERIFIED'),'evidence_level':'E1','runtime':rt,'load_wall_ns':load_ns,'capture_wall_ns':capture_ns,'compile_wall_ns':compile_ns,'config_sha256':fsha(config_path),'manifest':manifest,
  'MEASURED':{'build_state_count':xb.shape[0],'evaluation_state_count':xe.shape[0],'build_assignment_counts':build_counts,'queries':queries,'residual':{'oracle':residual_diagnostics(ye,rawq['eval_oracle_fp32']['candidate']),'router':residual_diagnostics(ye,rawq['eval_router_fp32']['candidate'])}},
  'DERIVED':{'target_resources':res,'minimum_service_tokens_to_amortize_build_forwards':math.ceil(xb.shape[0]/P50)},'gates':core['gates'],'authoritative_decision':decision,'deterministic_core_sha256':sha(canon(core)),
  'claim_boundary':{'official_checkpoint_loaded':True,'actual_causal_states':True,'disjoint_build_evaluation_prompts':True,'state_lookup_table':False,'oracle_deployable':False,'complete_mlp_replaced':False,'complete_layer':'NOT_TESTED','405b':'NOT_TESTED','8gib_gpu':'NOT_TESTED','latency':'NOT_TESTED'}}
 outdir.mkdir(parents=True,exist_ok=True); (outdir/'raw').mkdir(exist_ok=True); (outdir/'artifacts').mkdir(exist_ok=True)
 (outdir/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+'\n'); (outdir/'raw/trace_rows.json').write_text(json.dumps(traces,indent=2,sort_keys=True,ensure_ascii=False)+'\n'); (outdir/'raw/query_rows.json').write_text(json.dumps(queries,indent=2,sort_keys=True,ensure_ascii=False)+'\n'); (outdir/'artifacts/library_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n'); (outdir/'artifacts/deterministic_core.json').write_bytes(canon(core))
 paths=['result.json','raw/trace_rows.json','raw/query_rows.json','artifacts/library_manifest.json','artifacts/deterministic_core.json']; (outdir/'checksums.sha256').write_text('\n'.join(f'{fsha(outdir/p)}  {p}' for p in paths)+'\n')
 return result


def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
 try: r=execute(a.config,a.output_dir)
 except Exception as e:
  a.output_dir.mkdir(parents=True,exist_ok=True); r={'experiment':'EXP-086A','source_sha':os.getenv('GITHUB_SHA','LOCAL_UNVERIFIED'),'authoritative_decision':'INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION','error':f'{type(e).__name__}: {e}'}; (a.output_dir/'result.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 print(json.dumps(r,indent=2,sort_keys=True,ensure_ascii=False)); return 0

if __name__=='__main__': raise SystemExit(main())
