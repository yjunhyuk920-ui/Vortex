"""Explicit storage chart and failed native rotary inverse probe.

This codec changes representation, not the checkpoint. It pays full decode and
encode at every boundary and therefore is NOT a fast transformed-body algorithm.
"""
from __future__ import annotations
import torch
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from native_graph import cache_flat,raw_bytes,tensor_meta


class PackedCache(DynamicCache):
    def __init__(self,payload,schema):
        super().__init__()
        self.payload=payload;self.schema=schema
        self.logical_layers=len(schema)//2
        self.logical_length=schema[0]['shape'][-2] if schema else 0

    def get_seq_length(self,layer_idx=0):return self.logical_length

    def get_max_cache_shape(self,layer_idx=0):return -1

    def __len__(self):return self.logical_layers


def pack_cache(cache):
    flat=cache_flat(cache);pieces=[];schema=[];offset=0
    for t in flat:
        if t.device.type!='cpu' or t.dtype!=torch.bfloat16:
            raise ValueError('Frozen storage ABI is CPU BF16')
        if any(s<0 for s in t.stride()):raise ValueError('Negative strides not admitted')
        b=t.detach().contiguous().view(torch.uint8).reshape(-1)
        meta=tensor_meta(t);meta.update(offset=offset,nbytes=b.numel())
        schema.append(meta);pieces.append(b);offset+=b.numel()
    payload=torch.cat(pieces) if pieces else torch.empty(0,dtype=torch.uint8)
    return PackedCache(payload,schema)


def unpack_cache(cache):
    if type(cache) is not PackedCache:raise TypeError('Expected encoded cache')
    out=DynamicCache()
    for i in range(cache.logical_layers):
        out.append_new_layers(i)
        for j,field in enumerate(('keys','values')):
            s=cache.schema[2*i+j]
            b=cache.payload.narrow(0,s['offset'],s['nbytes'])
            logical=b.view(torch.bfloat16).reshape(s['shape'])
            # Preserve native shape/stride/offset, not merely values. Offsets are
            # charged to backing storage and needed for an exact layout guard.
            span=1+sum((d-1)*st for d,st in zip(s['shape'],s['stride']))+s['storage_offset']
            backing=torch.empty(span,dtype=torch.bfloat16)
            t=torch.as_strided(backing,s['shape'],s['stride'],s['storage_offset'])
            t.copy_(logical)
            setattr(out.layers[i],field,t)
    return out


class NativeStorageExecutor:
    def __init__(self,original):
        self.original=original;self.records=[]

    def forward(self,*args,**kwargs):
        if args:raise ValueError('Named input ABI only')
        cache=kwargs.get('past_key_values');decode_bytes=0
        if type(cache) is PackedCache:
            decode_bytes=cache.payload.numel();kwargs=dict(kwargs)
            kwargs['past_key_values']=unpack_cache(cache)
        elif cache is not None and type(cache) is not DynamicCache:
            raise ValueError('Unsupported cache class')
        out=self.original(**kwargs)
        packed=pack_cache(out.past_key_values)
        self.records.append({'original_full_forward_calls':1,'decode_payload_bytes':decode_bytes,
                             'encode_payload_bytes':packed.payload.numel(),
                             'minimum_codec_read_write_bytes':2*(decode_bytes+packed.payload.numel()),
                             'allocation_metadata_contiguous_copy_extra':True})
        return CausalLMOutputWithPast(logits=out.logits,past_key_values=packed,
                                     hidden_states=out.hidden_states,attentions=out.attentions)


def rotate_half(x):
    a,b=x.chunk(2,dim=-1);return torch.cat((-b,a),dim=-1)


def rotary_probe(model,original,prefix):
    captured={}
    def hook(label):
        def take(module,args,out):captured[label]=out.detach().clone()
        return take
    attn=model.model.layers[0].self_attn
    hooks=[attn.q_proj.register_forward_hook(hook('q')),attn.k_proj.register_forward_hook(hook('k'))]
    with torch.inference_mode():original(input_ids=prefix,use_cache=True,return_dict=True)
    for h in hooks:h.remove()
    q=captured['q'].reshape(1,4,9,64).transpose(1,2)
    k=captured['k'].reshape(1,4,3,64).transpose(1,2)
    rows=[]
    for position in [0,1,7,31]:
        pos=torch.full((1,4),position,dtype=torch.long)
        cos,sin=model.model.rotary_emb(captured['q'],pos)
        c=cos.unsqueeze(1);s=sin.unsqueeze(1)
        qrot=q*c+rotate_half(q)*s;krot=k*c+rotate_half(k)*s
        back=krot*c-rotate_half(krot)*s
        ub=k.contiguous().view(torch.int16);vb=back.contiguous().view(torch.int16)
        diff=(ub!=vb);idx=diff.flatten().nonzero().flatten()
        original_scores=torch.matmul(qrot,krot.repeat_interleave(3,dim=1).transpose(-1,-2))
        unrotated_scores=torch.matmul(q,k.repeat_interleave(3,dim=1).transpose(-1,-2))
        witness=None
        if idx.numel():
            j=int(idx[0]);witness={'flat_index':j,'raw_key':float(k.flatten()[j]),
                'rotated_key':float(krot.flatten()[j]),'inverse_key':float(back.flatten()[j]),
                'raw_word':int(ub.flatten()[j])&65535,'inverse_word':int(vb.flatten()[j])&65535}
        rows.append({'position':position,'key_coordinates':k.numel(),
                     'native_inverse_mismatches':int(diff.sum()),'witness':witness,
                     'score_coordinates':original_scores.numel(),
                     'native_orthogonal_score_mismatches':int((original_scores.view(torch.int16)!=unrotated_scores.view(torch.int16)).sum()),
                     'scope':'observed real keys/queries,synthetic repeated rotary position;not every chart rejected'})
    return {'rows':rows,'native_inverse_chart':'REJECTED_IF_ANY_MISMATCH',
            'does_not_reject_raw_key_preservation_or_all_state_encodings':True}
