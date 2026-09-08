from pathlib import Path
import hashlib,inspect,json,sys,platform
import torch,transformers
from transformers import AutoModelForCausalLM,DynamicCache
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
model=AutoModelForCausalLM.from_pretrained(ROOT/'.cache/model',torch_dtype='auto',local_files_only=True,trust_remote_code=False).eval()
def method_source(obj):
    try:return inspect.getsource(obj)
    except (TypeError,OSError):return 'SOURCE_UNAVAILABLE'
data={'torch':torch.__version__,'transformers':transformers.__version__,'python':sys.version,'platform':platform.platform(),
      'model_class':type(model).__name__,'dtype':str(model.dtype),'attention':model.config._attn_implementation,
      'config':model.config.to_dict(),'forward_signature':str(inspect.signature(model.forward)),
      'cache_signature':str(inspect.signature(DynamicCache)),
      'cache_methods':[x for x in ['to_legacy_cache','from_legacy_cache','get_seq_length'] if hasattr(DynamicCache,x)],
      'torch_build':torch.__config__.show()}
with torch.inference_mode():
    ids=torch.tensor([[1,42,73,11]],dtype=torch.long);out=model(input_ids=ids,use_cache=True,return_dict=True)
    cache=out.past_key_values
    data['smoke']={'logits_shape':list(out.logits.shape),'logits_dtype':str(out.logits.dtype),
                   'cache_type':type(cache).__name__,'cache_layers':len(cache),
                   'cache_seq_length':cache.get_seq_length(),
                   'cache_pair_shapes':[[list(t.shape) for t in pair] for pair in cache],
                   'cache_dict_keys':list(cache.__dict__),
                   'layer0_keys':list(cache.layers[0].__dict__) if hasattr(cache,'layers') else None,
                   'logits_sha256':hashlib.sha256(out.logits.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
    data['default_generation_config']=model.generation_config.to_dict()
path=ROOT/'results/reference_inspection.json';path.write_text(json.dumps(data,default=str,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
source_dir=ROOT/'results/reference_sources';source_dir.mkdir(exist_ok=True)
for label,method in [('model_forward',model.forward),('llama_forward',model.model.forward),('layer_forward',model.model.layers[0].forward),('attention_forward',model.model.layers[0].self_attn.forward),('rotary_forward',model.model.rotary_emb.forward),('cache_init',DynamicCache.__init__),('cache_update',DynamicCache.update),('generation_sample',model._sample)]:
    s=method_source(method);(source_dir/(label+'.txt')).write_text(s,encoding='utf-8')
    print(label,hashlib.sha256(s.encode()).hexdigest())
print(json.dumps({k:data[k] for k in ['torch','transformers','model_class','dtype','attention','forward_signature','cache_signature','cache_methods','smoke']},default=str,ensure_ascii=False),flush=True)
