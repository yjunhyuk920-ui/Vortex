from pathlib import Path
import json,traceback
import torch
from transformers import AutoModelForCausalLM
from native_graph import NativeGraphExecutor,raw_bytes,cache_flat,tensor_meta
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
model=AutoModelForCausalLM.from_pretrained(ROOT/'.cache/model',torch_dtype='auto',local_files_only=True).eval()
engine=NativeGraphExecutor(model,model.forward,ROOT/'results/probe_graph')
with torch.inference_mode():
    kw=dict(input_ids=torch.tensor([[1,42,73,11]]),use_cache=True,return_dict=True)
    ref=model(**kw);out=engine.forward(**kw)
    a=(ref.logits,*cache_flat(ref.past_key_values));b=(out.logits,*cache_flat(out.past_key_values))
    result={'roots':len(a),'bit_mismatches':sum(raw_bytes(x)!=raw_bytes(y) for x,y in zip(a,b)),
            'layout_mismatches':sum(tensor_meta(x)!=tensor_meta(y) for x,y in zip(a,b)),
            'graphs':[{'before':r['before']['calls'],'after':r['after']['calls'],
                       'matrix_before':r['before']['matrix_macs'],'matrix_after':r['after']['matrix_macs']} for r in engine.records]}
    print(json.dumps(result),flush=True)
    (ROOT/'results/probe_graph_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
