"""Actual CPU HF comparison for a restricted two-support BF16 producer."""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from scalar_producer import compile_store, project


def raw(t):
    t=t.detach().cpu().contiguous()
    return t.view(torch.uint16).numpy().tobytes() if t.dtype==torch.bfloat16 else t.numpy().tobytes()


def model_hash(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode()+str(tensor.dtype).encode()+str(list(tensor.shape)).encode()+raw(tensor))
    return digest.hexdigest()


def cache_words(cache,layers):
    return [[raw(cache.layers[i].keys).hex(),raw(cache.layers[i].values).hex()] for i in range(layers)]


def cache_state(value):
    """Explicit value/state serializer; pointers are not continuation state."""
    if isinstance(value,torch.Tensor):
        return {'dtype':str(value.dtype),'device':str(value.device),'shape':list(value.shape),
                'stride':list(value.stride()),'requires_grad':value.requires_grad,
                'layout':str(value.layout),'words':raw(value).hex()}
    if isinstance(value,type):
        return value.__module__+'.'+value.__qualname__
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    if isinstance(value,(list,tuple)):
        return [cache_state(item) for item in value]
    if isinstance(value,dict):
        return {key:cache_state(item) for key,item in sorted(value.items())}
    if type(value).__module__=='transformers.cache_utils':
        return {'class':type(value).__qualname__,'fields':cache_state(vars(value))}
    raise TypeError(f'unaccounted cache state type: {type(value)}')


@contextmanager
def sampling_trace(records):
    original=torch.multinomial
    def traced(probabilities,*args,**kwargs):
        before=raw(torch.random.get_rng_state()).hex()
        result=original(probabilities,*args,**kwargs)
        records.append({'probability_words':raw(probabilities).hex(),
                        'rng_before':before,'rng_after':raw(torch.random.get_rng_state()).hex(),
                        'sample':result.tolist()})
        return result
    torch.multinomial=traced
    try:
        yield
    finally:
        torch.multinomial=original


def build_fixture():
    n,m,layers,hd=32,16,2,2
    config=LlamaConfig(vocab_size=n,hidden_size=n,intermediate_size=n,
        num_hidden_layers=layers,num_attention_heads=n//hd,num_key_value_heads=m//hd,
        head_dim=hd,max_position_embeddings=64,tie_word_embeddings=False,
        attention_bias=False,mlp_bias=False,use_cache=True,eos_token_id=None,pad_token_id=0)
    model=LlamaForCausalLM(config).eval().to(torch.bfloat16)
    rng=random.Random(260909)
    matrices=[]
    for _ in range(layers):
        words=[]
        for index in range(m*n):
            sign=rng.getrandbits(1)
            exponent=rng.randint(-8,8)
            mantissa=rng.randrange(128)
            words.append((sign<<15) if index%37==0 else (sign<<15)|((exponent+127)<<7)|mantissa)
        matrices.append(torch.tensor(words,dtype=torch.uint16).view(torch.bfloat16).reshape(m,n))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for j in range(n):
            model.model.embed_tokens.weight[j,j]=1
            model.model.embed_tokens.weight[j,(j+1)%n]=-1 if j%2 else 1
        model.model.norm.weight.fill_(1)
        for i,layer in enumerate(model.model.layers):
            layer.input_layernorm.weight.fill_(1)
            layer.post_attention_layernorm.weight.fill_(1)
            layer.self_attn.v_proj.weight.copy_(matrices[i])
    return model,matrices


@dataclass
class Compiled:
    n:int
    m:int
    layers:int
    hd:int
    stores:list
    scales:list
    zero_key:torch.Tensor
    compile_report:dict


def compile_fixture(model):
    n=model.config.hidden_size
    layers=len(model.model.layers)
    hd=model.config.head_dim
    m=model.config.num_key_value_heads*hd
    parameters_checked=0
    with torch.inference_mode():
        expected=torch.zeros_like(model.model.embed_tokens.weight)
        for j in range(n):
            expected[j,j]=1
            expected[j,(j+1)%n]=-1 if j%2 else 1
        if raw(expected)!=raw(model.model.embed_tokens.weight):
            raise ValueError('outside fixed two-support embedding family')
        for name,param in model.named_parameters():
            parameters_checked+=param.numel()
            if name.endswith('v_proj.weight'):
                if not bool(torch.isfinite(param).all()) or not bool((param.abs()<=512).all()):
                    raise ValueError('outside bounded finite v_proj family')
            elif name=='model.embed_tokens.weight':
                pass
            elif 'layernorm.weight' in name or name=='model.norm.weight':
                if not bool((param==1).all()):
                    raise ValueError('normalization weights must be one')
            elif bool((param!=0).any()) or bool(torch.signbit(param).any()):
                raise ValueError('all other checkpoint weights must be +0')
        stores=[]
        scales=[]
        norm_words_checked=0
        for layer in model.model.layers:
            words=layer.self_attn.v_proj.weight.detach().view(torch.uint16).tolist()
            stores.append(compile_store(words))
            normalized=layer.input_layernorm(model.model.embed_tokens.weight.unsqueeze(0))[0]
            scale=int(normalized[0,0].view(torch.uint16).item())
            for j in range(n):
                predicted=torch.zeros(n,dtype=torch.uint16)
                predicted[j]=scale
                predicted[(j+1)%n]=scale ^ (0x8000 if j%2 else 0)
                assert raw(predicted.view(torch.bfloat16))==raw(normalized[j])
                norm_words_checked+=n
            scales.append(scale)
        length=model.config.max_position_embeddings
        positions=torch.arange(length,dtype=torch.long).unsqueeze(0)
        probe=torch.zeros((1,length,n),dtype=torch.bfloat16)
        cos,sin=model.model.rotary_emb(probe,positions)
        zero=torch.zeros((1,1,length,hd),dtype=torch.bfloat16)
        _,key=apply_rotary_pos_emb(zero,zero,cos,sin)
        template=key[0,0].contiguous().clone()
    return Compiled(n,m,layers,hd,stores,scales,template,
        {'checkpoint_parameters_scanned':parameters_checked,'norm_words_verified':norm_words_checked,
         'vproj_words_read_and_transposed':layers*m*n,'original_and_compiled_vproj_bytes':4*layers*m*n,
         'zero_sign_count_payload_bytes':4*layers*m,'scale_bytes':2*layers,
         'zero_key_template_bytes':2*length*hd,
         'extra_paid':'full checkpoint hash/validation, native norm/RoPE initialization, allocations, Python objects, control and code'})


def compiled_step(compiled,token,cache=None):
    if not 0<=token<compiled.n:
        raise ValueError('illegal token')
    if cache is None:
        cache=DynamicCache()
    position=cache.get_seq_length()
    if position>=compiled.zero_key.shape[0]:
        raise ValueError('outside compiled legal position range')
    stats=[]
    for i in range(compiled.layers):
        x=[0]*compiled.n
        x[token]=compiled.scales[i]
        x[(token+1)%compiled.n]=compiled.scales[i] ^ (0x8000 if token%2 else 0)
        values,counts=project(compiled.stores[i],x,max_support=2)
        vector=torch.tensor(values,dtype=torch.uint16).view(torch.bfloat16)
        v=vector.reshape(1,compiled.m//compiled.hd,1,compiled.hd)
        k=compiled.zero_key[position].reshape(1,1,1,compiled.hd).expand(1,compiled.m//compiled.hd,1,compiled.hd).contiguous()
        cache.update(k,v,i)
        stats.append(counts)
    return torch.zeros((1,1,compiled.n),dtype=torch.bfloat16),cache,stats


@contextmanager
def prohibit_dense_forward(model):
    linear_forward=torch.nn.Linear.forward
    original_forward=model.forward
    def forbidden(*args,**kwargs):
        raise AssertionError('dense/model forward invoked by compiled runtime')
    torch.nn.Linear.forward=forbidden
    model.forward=forbidden
    try:
        yield
    finally:
        model.forward=original_forward
        torch.nn.Linear.forward=linear_forward


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit('refuse to overwrite evidence')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(260909)
    model,matrices=build_fixture()
    manifest={'config':model.config.to_dict(),'torch':torch.__version__,
        'transformers':transformers.__version__,'python':sys.version,
        'attention_implementation':model.config._attn_implementation,
        'device':'cpu','dtype':'torch.bfloat16','threads':torch.get_num_threads(),
        'interop_threads':torch.get_num_interop_threads(),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'mkldnn_enabled':torch.backends.mkldnn.enabled,'torch_build':torch.__config__.show(),
        'seed':260909,'token_input':'direct integer IDs; no tokenizer',
        'source_generator':'PREREGISTRATION.md frozen Python Random source',
        'supported_template_positions':64,'cache_class':'transformers.cache_utils.DynamicCache'}
    (args.output_dir/'manifest.json').write_bytes((json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode())
    checkpoint_before=model_hash(model)
    compiled=compile_fixture(model)
    tokens=list(range(32))+[31,0,17,17,3,25,0,31]
    native_cache=None
    compiled_cache=None
    records=[]
    first_failure=None
    cache_metadata_exact=True
    rng_before=raw(torch.random.get_rng_state())
    with torch.inference_mode():
        full=model(input_ids=torch.tensor([tokens]),attention_mask=torch.ones((1,len(tokens)),dtype=torch.long),use_cache=True,return_dict=True)
        full_cache=cache_words(full.past_key_values,compiled.layers)
        for step,token in enumerate(tokens):
            out=model(input_ids=torch.tensor([[token]]),past_key_values=native_cache,use_cache=True,return_dict=True)
            native_cache=out.past_key_values
            with prohibit_dense_forward(model):
                logits,compiled_cache,stats=compiled_step(compiled,token,compiled_cache)
            native_words=cache_words(native_cache,compiled.layers)
            candidate_words=cache_words(compiled_cache,compiled.layers)
            state_equal=cache_state(native_cache)==cache_state(compiled_cache)
            cache_metadata_exact=cache_metadata_exact and state_equal
            equality=(out.logits.dtype==logits.dtype and raw(out.logits)==raw(logits) and native_words==candidate_words and state_equal)
            records.append({'step':step,'token':token,'exact':equality,'cache_all_fields_exact':state_equal,
                'native_logits_dtype':str(out.logits.dtype),'counts':stats})
            if not equality:
                first_failure={'step':step,'token':token,'native_logits':raw(out.logits).hex(),'compiled_logits':raw(logits).hex(),
                    'native_cache':cache_state(native_cache),'compiled_cache':cache_state(compiled_cache)}
                break
    deterministic_rng_unchanged=raw(torch.random.get_rng_state())==rng_before
    full_incremental_equal=full_cache==cache_words(native_cache,compiled.layers) if first_failure is None else False
    generation={}
    if first_failure is None:
        prefix=[0,7,31]
        native_samples=[]
        compiled_samples=[]
        sample_rng=torch.random.get_rng_state().clone()
        with torch.inference_mode(),sampling_trace(native_samples):
            reference=model.generate(input_ids=torch.tensor([prefix]),attention_mask=torch.ones((1,len(prefix)),dtype=torch.long),
                do_sample=True,top_k=0,top_p=1.0,temperature=1.0,eos_token_id=None,pad_token_id=0,
                max_new_tokens=8,return_dict_in_generate=True,output_scores=True,output_logits=True,use_cache=True)
        native_rng_after=raw(torch.random.get_rng_state())
        native_sample_cache=cache_words(reference.past_key_values,compiled.layers)
        native_sample_state=cache_state(reference.past_key_values)
        torch.random.set_rng_state(sample_rng)
        sampled=list(prefix)
        cache=None
        with torch.inference_mode(),prohibit_dense_forward(model),sampling_trace(compiled_samples):
            for token in prefix:
                logits,cache,_=compiled_step(compiled,token,cache)
            for step in range(8):
                scores=logits[:,-1,:].clone().float()
                probability=torch.nn.functional.softmax(scores,dim=-1)
                token=int(torch.multinomial(probability,num_samples=1).squeeze(1).item())
                sampled.append(token)
                if step<7:
                    logits,cache,_=compiled_step(compiled,token,cache)
        generation={'native_tokens':reference.sequences[0].tolist(),'compiled_tokens':sampled,
            'tokens_exact':reference.sequences[0].tolist()==sampled,
            'RNG_consumption_and_next_state_exact':native_rng_after==raw(torch.random.get_rng_state()),
            'final_cache_exact':native_sample_cache==cache_words(cache,compiled.layers),
            'final_cache_all_fields_exact':native_sample_state==cache_state(cache),
            'each_sample_logits_probability_RNG_exact':len(native_samples)==8 and native_samples==compiled_samples,
            'native_generate_executed':True,'compiled_dense_forward_prohibited':True}
        (args.output_dir/'sampling_records.json').write_bytes((json.dumps({'native':native_samples,'compiled':compiled_samples},indent=2,sort_keys=True)+'\n').encode())
    result={'kind':'E1_RESTRICTED_TWO_SUPPORT_BF16_CAUSAL_PRODUCER','torch':torch.__version__,
        'transformers':transformers.__version__,'device':'cpu','threads':torch.get_num_threads(),
        'checkpoint_sha256_before':checkpoint_before,'checkpoint_sha256_after':model_hash(model),
        'checkpoint_unchanged':checkpoint_before==model_hash(model),'compile_cost':compiled.compile_report,
        'manual_steps':len(records),'manual_all_logits_KV_exact':first_failure is None,
        'cache_all_fields_including_strides_exact':cache_metadata_exact,
        'native_full_prefill_incremental_cache_exact':full_incremental_equal,
        'deterministic_native_and_compiled_RNG_unchanged':deterministic_rng_unchanged,
        'generation':generation,'first_failure':first_failure,
        'runtime_cost_scope':{'vproj_source_words_per_token':compiled.layers*compiled.m*2,
          'row_metadata_words_per_token':compiled.layers*compiled.m,'input_words_scanned_per_token':compiled.layers*compiled.n,
          'native_full_vproj_words_per_token':compiled.layers*compiled.m*compiled.n,
          'new_KV_payload_bytes_per_token':4*compiled.layers*compiled.m,
          'DynamicCache_update_existing_KV_copy_bytes_at_prior_T':f'{4*compiled.layers*compiled.m} * T',
          'extra_paid':'key-template reads/broadcast, tensor conversion, logits/sampler/RNG, cache copies, Python/control/allocator, full source storage and compilation'},
        'source_domain_HF':'generated bounded BF16 source [-8,8] exponents with fixed two-support embeddings and zero remaining dense paths',
        'scalar_domain_is_broader_than_HF_fixture':True,'arbitrary_checkpoint_executor':False,
        'arbitrary_dense_hidden_state':False,'target_hardware_tested':False,
        'THEORY_STATUS':'NOT_ESTABLISHED','HARDWARE_STATUS':'NOT_TESTED','CORE_ADMISSION':False}
    for name,value in [('summary.json',result),('manual_records.json',records)]:
        (args.output_dir/name).write_bytes((json.dumps(value,indent=2,sort_keys=True)+'\n').encode())
    paths=sorted(args.output_dir.iterdir())
    (args.output_dir/'checksums.sha256').write_bytes(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in paths).encode())
    print(json.dumps({k:result[k] for k in ['manual_steps','manual_all_logits_KV_exact','native_full_prefill_incremental_cache_exact','deterministic_native_and_compiled_RNG_unchanged','generation','checkpoint_unchanged']}))
    if first_failure is not None or not full_incremental_equal or not deterministic_rng_unchanged or (generation and not all(generation[k] for k in ['tokens_exact','RNG_consumption_and_next_state_exact','final_cache_exact','final_cache_all_fields_exact','each_sample_logits_probability_RNG_exact'])):
        raise SystemExit('restricted producer did not pass; failure evidence retained')


if __name__=='__main__':
    main()
