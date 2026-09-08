"""Guarded, finite native functional-relation constructor. No kernel is free.

The saved program refers to original checkpoint tensors by name. It does not
contain a truth table, future tokens, or copied checkpoint weights.
"""
from __future__ import annotations
import collections, hashlib, json, operator, time
from pathlib import Path
import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.experimental.proxy_tensor import make_fx
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast


def raw_bytes(t):
    return t.detach().contiguous().view(torch.uint8).numpy().tobytes()


def tensor_meta(t):
    return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                device=str(t.device), storage_offset=t.storage_offset())


def cache_flat(cache):
    if cache is None or cache.get_seq_length() == 0:
        return ()
    if type(cache) is not DynamicCache or cache.cache_processor is not None:
        raise ValueError('Only unprocessed DynamicCache is in this guarded ABI')
    return tuple(t for pair in cache.to_legacy_cache() for t in pair)


def native_cache(flat):
    if not flat:
        return DynamicCache()
    return DynamicCache.from_legacy_cache(tuple(zip(flat[::2], flat[1::2])))


def resolve_attr(obj, name):
    for part in name.split('.'):
        obj = getattr(obj, part)
    return obj


def encode(x):
    if isinstance(x, Node): return {'node': x.name}
    if isinstance(x, tuple): return {'tuple': [encode(a) for a in x]}
    if isinstance(x, list): return {'list': [encode(a) for a in x]}
    if isinstance(x, dict): return {'dict': [[encode(k), encode(v)] for k,v in x.items()]}
    if isinstance(x, slice): return {'slice': [encode(x.start),encode(x.stop),encode(x.step)]}
    if isinstance(x, torch.dtype): return {'dtype': str(x).split('.')[-1]}
    if isinstance(x, torch.device): return {'device': str(x)}
    if isinstance(x, torch.layout): return {'layout': str(x).split('.')[-1]}
    if isinstance(x, torch.memory_format): return {'memory_format': str(x).split('.')[-1]}
    if isinstance(x, float): return {'float': x.hex()}
    if x is None or isinstance(x, (int,bool,str)): return x
    raise TypeError(f'Unsupported IR literal: {type(x)} {x}')


def decode(x, nodes):
    if not isinstance(x,dict): return x
    if 'node' in x: return nodes[x['node']]
    if 'tuple' in x: return tuple(decode(a,nodes) for a in x['tuple'])
    if 'list' in x: return [decode(a,nodes) for a in x['list']]
    if 'dict' in x: return {decode(k,nodes):decode(v,nodes) for k,v in x['dict']}
    if 'slice' in x: return slice(*(decode(a,nodes) for a in x['slice']))
    if 'dtype' in x: return getattr(torch,x['dtype'])
    if 'layout' in x: return getattr(torch,x['layout'])
    if 'memory_format' in x: return getattr(torch,x['memory_format'])
    if 'device' in x: return torch.device(x['device'])
    if 'float' in x: return float.fromhex(x['float'])
    raise ValueError('Unknown encoded field')


def target_name(t):
    if t is operator.getitem: return 'operator.getitem'
    if isinstance(t, torch._ops.OpOverload): return str(t)
    raise TypeError(f'Unsupported callable: {t}')


def target_load(s):
    if s == 'operator.getitem': return operator.getitem
    parts=s.split('.')
    if len(parts)!=3 or parts[0]!='aten': raise ValueError('Not an allowed ATen target')
    return getattr(getattr(torch.ops.aten,parts[1]),parts[2])


def proof_guards(gm):
    unsupported=[]
    for n in gm.graph.nodes:
        if n.op!='call_function': continue
        target_name(n.target)
        schema=getattr(n.target,'_schema',None)
        if schema is not None and schema.is_mutable:
            unsupported.append((n.name,'mutable',str(n.target)))
        if torch.Tag.nondeterministic_seeded in getattr(n.target,'tags',()):
            unsupported.append((n.name,'random',str(n.target)))
    if unsupported: raise ValueError(f'Guard rejects effects: {unsupported[:8]}')
    return {'no_mutable_native_ops':True,'no_seeded_random_ops':True,
            'fixed_shape_dtype_stride':True,'frozen_checkpoint_required':True,
            'unknown_native_ops_retained_and_charged':True}


def pure_cse_dce(gm):
    """Never reorder arithmetic. Only identical read-only expressions can merge.

    CSE excludes output roots and any alias-producing path leading to a root, so
    it cannot merge distinct externally returned allocations via an output view.
    """
    proof_guards(gm)
    roots=set()
    torch.fx.map_arg(next(n for n in gm.graph.nodes if n.op=='output').args,
                     lambda n: roots.add(n))
    stack=list(roots); protected=set(roots)
    while stack:
        n=stack.pop()
        schema=getattr(n.target,'_schema',None) if n.op=='call_function' else None
        is_alias=n.target is operator.getitem if n.op=='call_function' else False
        is_alias=is_alias or (schema is not None and any(r.alias_info for r in schema.returns))
        if is_alias:
            for p in n.all_input_nodes:
                if p not in protected: protected.add(p); stack.append(p)
    allowed={'aten.add.Tensor','aten.sub.Tensor','aten.mul.Tensor','aten.neg.default',
             'aten.mm.default','aten.bmm.default','aten.silu.default','aten.cos.default',
             'aten.sin.default','aten.rsqrt.default','aten.mean.dim','aten.pow.Tensor_Scalar'}
    seen={};merged=[]
    for n in list(gm.graph.nodes):
        if n.op!='call_function' or str(n.target) not in allowed or n in protected: continue
        schema=n.target._schema
        if any(r.alias_info for r in schema.returns): continue
        key=json.dumps([str(n.target),encode(n.args),encode(n.kwargs)],sort_keys=True)
        if key in seen:
            old=seen[key];merged.append({'removed':n.name,'retained':old.name,'op':str(n.target)})
            n.replace_all_uses_with(old);gm.graph.erase_node(n)
        else: seen[key]=n
    # With effects explicitly rejected, root reachability is sufficient here.
    live=set()
    stack=[n for n in gm.graph.nodes if n.op=='output']
    while stack:
        n=stack.pop()
        if n in live: continue
        live.add(n);stack.extend(n.all_input_nodes)
    dead=[]
    for n in reversed(list(gm.graph.nodes)):
        if n not in live and n.op not in ('placeholder','output'):
            dead.append({'name':n.name,'op':str(n.target)});gm.graph.erase_node(n)
    gm.graph.lint();gm.recompile()
    return dict(cse=merged,dead=dead,guards=proof_guards(gm))


def meta_values(n):
    v=n.meta.get('val')
    if isinstance(v,torch.Tensor): return [v]
    if isinstance(v,(tuple,list)): return [t for t in v if isinstance(t,torch.Tensor)]
    return []


def work_inventory(gm):
    nodes=list(gm.graph.nodes);calls=[n for n in nodes if n.op=='call_function']
    hist=collections.Counter(str(n.target) for n in calls)
    mats=[];macs=0;operand_bytes=0;sdpa=[]
    for n in calls:
        op=str(n.target)
        if op in ('aten.mm.default','aten.bmm.default','aten.addmm.default'):
            off=1 if op=='aten.addmm.default' else 0
            a=n.args[off].meta.get('val');b=n.args[off+1].meta.get('val')
            if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor): continue
            cnt=a.shape[-2]*a.shape[-1]*b.shape[-1]
            if a.ndim==3:cnt*=a.shape[0]
            nb=b.numel()*b.element_size();macs+=cnt;operand_bytes+=nb
            mats.append(dict(node=n.name,op=op,a=list(a.shape),b=list(b.shape),macs=int(cnt),rhs_bytes=nb))
        if 'scaled_dot_product' in op:
            q,k,v=(arg.meta['val'] for arg in n.args[:3])
            cnt=2*q.shape[0]*q.shape[1]*q.shape[-2]*k.shape[-2]*q.shape[-1]
            sdpa.append(dict(node=n.name,op=op,q=list(q.shape),k=list(k.shape),v=list(v.shape),
                             qk_pv_macs=int(cnt),softmax_and_backend_workspace_extra=True))
    # Exact last-use schedule in this IR, allocating each result separately is a
    # conservative logical bound, not a measured allocator/native workspace peak.
    last={n:i for i,n in enumerate(nodes)}
    for i,n in enumerate(nodes):
        for inp in n.all_input_nodes:last[inp]=max(last[inp],i)
    sizes={n:sum(t.numel()*t.element_size() for t in meta_values(n)) for n in nodes}
    active={};peak=0;result_bytes=0
    for i,n in enumerate(nodes):
        if n.op=='call_function':active[n]=sizes[n];result_bytes+=sizes[n]
        peak=max(peak,sum(active.values()))
        for p in list(active):
            if last[p]<=i:del active[p]
    output=next(n for n in nodes if n.op=='output')
    roots=[];torch.fx.map_arg(output.args,lambda n:roots.append(n.name))
    return dict(nodes=len(nodes),calls=len(calls),operator_histogram=dict(sorted(hist.items())),
                matrix_calls=len(mats),matrix_macs=int(macs),matrix_rhs_payload_bytes=int(operand_bytes),
                matrices=mats,attention=sdpa,root_names=roots,
                logical_native_result_bytes=int(result_bytes),logical_last_use_peak_bytes=int(peak),
                physical_memory_traffic='NOT_MEASURED',native_workspace_and_allocator_extra=True)


def export_program(gm,model,path):
    params={**dict(model.named_parameters()),**dict(model.named_buffers())}
    records=[];attributes={}
    for n in gm.graph.nodes:
        rec={'name':n.name,'op':n.op,'args':encode(n.args),'kwargs':encode(n.kwargs)}
        if n.op=='get_attr':
            v=resolve_attr(gm,n.target)
            if not isinstance(v,torch.Tensor):raise TypeError('Non-tensor constant')
            hit=next((k for k,t in params.items() if t.data_ptr()==v.data_ptr() and
                      tensor_meta(t)==tensor_meta(v)),None)
            info={'meta':tensor_meta(v)}
            if hit is not None:info.update(kind='checkpoint',name=hit)
            else:
                data=raw_bytes(v)
                if len(data)>4096:raise ValueError('Large unbound constant may hide captured input/state')
                info.update(kind='literal',hex=data.hex())
            attributes[n.target]=info;rec['target']=n.target
        elif n.op=='call_function':rec['target']=target_name(n.target)
        elif n.op=='placeholder':rec['target']=n.target
        elif n.op=='output':rec['target']='output'
        else:raise TypeError(f'Unsupported node kind {n.op}')
        records.append(rec)
    data={'format':'VORTEX_NATIVE_DAG_V1','nodes':records,'attributes':attributes,
          'torch':torch.__version__,'model_weights_embedded':False}
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    return data


def load_program(path,model):
    data=json.loads(path.read_text(encoding='utf-8'))
    if data['format']!='VORTEX_NATIVE_DAG_V1' or data['torch']!=torch.__version__:
        raise ValueError('Native code ABI mismatch')
    holder=torch.nn.Module();nodes={};g=Graph()
    params={**dict(model.named_parameters()),**dict(model.named_buffers())}
    for name,info in data['attributes'].items():
        if '.' in name:raise ValueError('Nested attribute not supported')
        if info['kind']=='checkpoint':t=params[info['name']]
        else:
            b=torch.tensor(list(bytes.fromhex(info['hex'])),dtype=torch.uint8)
            dtype=getattr(torch,info['meta']['dtype'].split('.')[-1])
            logical=b.view(dtype).reshape(info['meta']['shape'])
            t=torch.empty_strided(info['meta']['shape'],info['meta']['stride'],dtype=dtype)
            t.copy_(logical)
        if list(t.shape)!=info['meta']['shape'] or list(t.stride())!=info['meta']['stride']:
            raise ValueError('Checkpoint layout mismatch')
        holder.register_buffer(name,t)
    for rec in data['nodes']:
        target=target_load(rec['target']) if rec['op']=='call_function' else rec['target']
        nodes[rec['name']]=g.create_node(rec['op'],target,decode(rec['args'],nodes),
                                      decode(rec['kwargs'],nodes),name=rec['name'])
    g.lint();return GraphModule(holder,g)


class NativeGraphExecutor:
    def __init__(self,model,original_forward,outdir):
        self.model=model;self.original=original_forward;self.outdir=Path(outdir)
        self.outdir.mkdir(parents=True,exist_ok=True);self.programs={};self.records=[]

    def forward(self,*args,**kwargs):
        if args:raise ValueError('Frozen wrapper requires named inputs')
        if kwargs.get('inputs_embeds') is not None:raise ValueError('inputs_embeds not in frozen population')
        if kwargs.get('output_hidden_states') or kwargs.get('output_attentions') or kwargs.get('labels') is not None:
            raise ValueError('Extra observer roots require separate capture')
        if kwargs.get('use_cache',True) is not True:raise ValueError('use_cache must be true')
        if kwargs.get('return_dict',True) is not True:raise ValueError('return_dict must be true')
        cache=kwargs.get('past_key_values');flat=cache_flat(cache)
        tensors={k:v for k,v in kwargs.items() if isinstance(v,torch.Tensor)}
        const={k:v for k,v in kwargs.items() if k not in tensors and k!='past_key_values'}
        const.setdefault('use_cache',True);const.setdefault('return_dict',True)
        names=tuple(sorted(tensors));inputs=tuple(tensors[n] for n in names)+flat
        sig={'names':names,'meta':[tensor_meta(t) for t in inputs],'const':encode(const),
             'past_layers':len(flat)//2}
        key=json.dumps(sig,sort_keys=True)
        if key not in self.programs:
            def f(*ins):
                kw=dict(const);kw.update(zip(names,ins[:len(names)]))
                kw['past_key_values']=native_cache(ins[len(names):])
                out=self.original(**kw)
                return (out.logits,*cache_flat(out.past_key_values))
            start=time.perf_counter();rng=torch.random.get_rng_state().clone()
            gm=make_fx(f,tracing_mode='real')(*inputs)
            if not torch.equal(rng,torch.random.get_rng_state()):
                raise ValueError('Capture consumed runtime RNG; reject rather than hide it')
            before=work_inventory(gm)
            idx=len(self.programs);(self.outdir/f'graph_{idx}_before.txt').write_text(gm.code,encoding='utf-8')
            transform=pure_cse_dce(gm);after=work_inventory(gm)
            path=self.outdir/f'graph_{idx}.json';export_program(gm,self.model,path)
            loaded=load_program(path,self.model)
            (self.outdir/f'graph_{idx}_after.txt').write_text(loaded.code,encoding='utf-8')
            rec=dict(index=idx,signature=sig,before=before,after=after,transform=transform,
                     build_seconds=time.perf_counter()-start,serialized_program_bytes=path.stat().st_size,
                     capture_original_forward_calls=1,checkpoint_load_and_graph_execution_extra=True)
            (self.outdir/f'graph_{idx}_inventory.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n',encoding='utf-8')
            self.records.append(rec);self.programs[key]=loaded
        outputs=self.programs[key](*inputs)
        if len(outputs)!=1+2*self.model.config.num_hidden_layers:raise RuntimeError('Lost state root')
        if cache is None:cache=DynamicCache()
        for i,(k,v) in enumerate(zip(outputs[1::2],outputs[2::2])):
            cache.append_new_layers(i)
            cache.layers[i].keys=k;cache.layers[i].values=v
        return CausalLMOutputWithPast(logits=outputs[0],past_key_values=cache)
