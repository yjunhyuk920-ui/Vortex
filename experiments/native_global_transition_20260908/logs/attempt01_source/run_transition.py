from __future__ import annotations
from pathlib import Path
import argparse,functools,hashlib,json,platform,time,sys
import torch,transformers
from transformers import AutoModelForCausalLM
from native_graph import NativeGraphExecutor,raw_bytes,cache_flat,tensor_meta
from native_state import NativeStorageExecutor,PackedCache,pack_cache,unpack_cache,rotary_probe

ROOT=Path(__file__).resolve().parent
PREFIXES=[[1,42,73,11],[1,314,271,18],[2,77,29,3]]
SEEDS=[260908,260909,260910]


def save_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n',encoding='utf-8')


def tensor_record(t,path):
    b=raw_bytes(t);path.write_bytes(b)
    return {**tensor_meta(t),'file':path.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def observe(out,path):
    path.mkdir(parents=True,exist_ok=True)
    cache=out.past_key_values
    observer_decode=type(cache) is PackedCache
    if observer_decode:cache=unpack_cache(cache)
    tensors=(out.logits,*cache_flat(cache))
    info={'logits':tensor_record(tensors[0],path/'logits.bin'),
          'kv':[tensor_record(t,path/f'kv_{i:02}.bin') for i,t in enumerate(tensors[1:])],
          'cache_length':cache.get_seq_length(),'cache_layers':len(cache),
          'cache_fields':list(cache.__dict__),
          'layer_fields':[list(layer.__dict__) for layer in cache.layers],
          'observer_decoded_packed_state':observer_decode}
    save_json(path/'state.json',info)
    return info


def compare_observations(a,b):
    ta=[a['logits'],*a['kv']];tb=[b['logits'],*b['kv']]
    if len(ta)!=len(tb):raise AssertionError('Different live root count')
    mismatch=[]
    for i,(x,y) in enumerate(zip(ta,tb)):
        for key in ['sha256','shape','stride','dtype','storage_offset','device']:
            if x[key]!=y[key]:mismatch.append({'root':i,'field':key,'a':x[key],'b':y[key]})
    for key in ['cache_length','cache_layers','cache_fields','layer_fields']:
        if a[key]!=b[key]:mismatch.append({'field':key,'a':a[key],'b':b[key]})
    return mismatch


def arm_run(model,original,executor,root,arm,prefix,seed):
    dest=root/f'{arm}_{seed}';dest.mkdir(parents=True,exist_ok=True);steps=[]
    call=original if executor is None else executor.forward
    @functools.wraps(original)
    def wrapped(*args,**kwargs):
        i=len(steps);d=dest/f'step_{i}';d.mkdir(exist_ok=True)
        pre=torch.random.get_rng_state().clone()
        info={'input_keys':sorted(kwargs),'tensor_inputs':{},'scalar_inputs':{},
              'rng_before':tensor_record(pre,d/'rng_before.bin')}
        for key,value in kwargs.items():
            if isinstance(value,torch.Tensor):info['tensor_inputs'][key]=tensor_record(value,d/f'in_{key}.bin')
            elif key!='past_key_values':info['scalar_inputs'][key]=str(value)
        out=call(*args,**kwargs)
        info['rng_after_forward']=tensor_record(torch.random.get_rng_state(),d/'rng_after_forward.bin')
        info['observation']=observe(out,d)
        steps.append(info);save_json(d/'record.json',info)
        return out
    model.forward=wrapped
    try:
        torch.manual_seed(seed)
        with torch.inference_mode():
            output=model.generate(input_ids=torch.tensor([prefix],dtype=torch.long),
                do_sample=True,top_k=0,top_p=1,temperature=1.0,max_new_tokens=4,
                use_cache=True,return_dict_in_generate=True,output_logits=True)
        finalrng=tensor_record(torch.random.get_rng_state(),dest/'rng_final.bin')
        seq=tensor_record(output.sequences,dest/'sequence.bin')
        report={'arm':arm,'seed':seed,'prefix':prefix,'steps':steps,
                'sequence':output.sequences.tolist(),'sequence_data':seq,'final_rng':finalrng,
                'own_logits_and_native_HF_sampler':True,'eos_not_overridden':True}
        save_json(dest/'run.json',report);return report
    finally:model.forward=original


def compare_runs(ref,cand):
    bad=[]
    if ref['sequence']!=cand['sequence']:bad.append('sequence')
    if ref['final_rng']['sha256']!=cand['final_rng']['sha256']:bad.append('final_rng')
    if len(ref['steps'])!=len(cand['steps']):bad.append('step_count')
    for i,(r,c) in enumerate(zip(ref['steps'],cand['steps'])):
        d=compare_observations(r['observation'],c['observation'])
        if d:bad.append({'step':i,'roots':d})
        for key in ['rng_before','rng_after_forward']:
            if r[key]['sha256']!=c[key]['sha256']:bad.append({'step':i,'rng':key})
        for key,info in r['tensor_inputs'].items():
            if info['sha256']!=c['tensor_inputs'][key]['sha256']:bad.append({'step':i,'input':key})
    return bad


def run(outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(ROOT/'.cache/model',torch_dtype='auto',local_files_only=True,trust_remote_code=False).eval()
    load_seconds=time.perf_counter()-start
    original=model.forward
    def paramhash():
        h=hashlib.sha256()
        for name,t in model.named_parameters():h.update(name.encode());h.update(raw_bytes(t))
        return h.hexdigest()
    weights_before=paramhash()
    graph=NativeGraphExecutor(model,original,outdir/'programs')
    storage=NativeStorageExecutor(original)
    # Entire untruncated prefill interface, separately from generate(logits_to_keep=1).
    with torch.inference_mode():
        kw=dict(input_ids=torch.tensor([PREFIXES[0]]),use_cache=True,return_dict=True)
        reference=observe(original(**kw),outdir/'full_prefill_reference')
        candidate=observe(graph.forward(**kw),outdir/'full_prefill_graph')
        prefill_errors=compare_observations(reference,candidate)
    runs=[];comparisons=[]
    for prefix,seed in zip(PREFIXES,SEEDS):
        for arm,ex in [('reference',None),('graph',graph),('packed_state',storage)]:
            rr=arm_run(model,original,ex,outdir,arm,prefix,seed);runs.append(rr)
            print(arm,seed,rr['sequence'],len(rr['steps']),flush=True)
        ref,g,c=runs[-3:]
        comparisons.append({'seed':seed,'graph_errors':compare_runs(ref,g),
                            'packed_state_errors':compare_runs(ref,c)})
    with torch.inference_mode():rotary=rotary_probe(model,original,torch.tensor([PREFIXES[0]]))
    save_json(outdir/'rotary_probe.json',rotary)
    ref_runs=[r for r in runs if r['arm']=='reference']
    ncoords=sum(r['steps'][i]['observation']['logits']['bytes']//2 for r in ref_runs for i in range(len(r['steps'])))
    kvcoords=sum(sum(t['bytes']//2 for t in step['observation']['kv']) for r in ref_runs for step in r['steps'])
    graph_summary=[]
    for r in graph.records:
        graph_summary.append({k:r[k] for k in ['index','signature','build_seconds','serialized_program_bytes','capture_original_forward_calls']})
        graph_summary[-1].update({key:{axis:r[key][axis] for axis in ['nodes','calls','matrix_calls','matrix_macs','matrix_rhs_payload_bytes','logical_native_result_bytes','logical_last_use_peak_bytes']} for key in ['before','after']})
        graph_summary[-1]['removed_cse']=len(r['transform']['cse']);graph_summary[-1]['removed_dead']=len(r['transform']['dead'])
    result={'source_pin':'HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2',
            'torch':torch.__version__,'transformers':transformers.__version__,'platform':platform.platform(),
            'load_seconds_observed':load_seconds,'weights_before':weights_before,'weights_after':paramhash(),
            'weights_unchanged':weights_before==paramhash(),'prefill_errors':prefill_errors,
            'comparisons':comparisons,'reference_decode_forward_steps':sum(len(r['steps']) for r in ref_runs),
            'reference_generation_logit_coordinates':ncoords,'reference_kv_coordinates':kvcoords,
            'full_prefill_logit_coordinates':reference['logits']['bytes']//2,
            'graph_programs':graph_summary,'storage_records':storage.records,
            'total_storage_codec_minimum_read_write_bytes':sum(r['minimum_codec_read_write_bytes'] for r in storage.records),
            'proof_scope':'guarded pinned tensor semantics, finite observed sequences; arbitrary length/CUDA/other checkpoints OPEN',
            'target_hardware':'NOT_TESTED','core_admission':False}
    save_json(outdir/'summary.json',result)
    assert not prefill_errors and all(not r['graph_errors'] and not r['packed_state_errors'] for r in comparisons)
    assert result['weights_unchanged']
    print(json.dumps({k:v for k,v in result.items() if k not in ('graph_programs','storage_records')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'results/run'))
    run(p.parse_args().out)
