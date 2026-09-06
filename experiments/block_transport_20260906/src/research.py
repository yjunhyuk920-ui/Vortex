"""Whole-block transport constructions and explicit rejection gates.

This is a reference research harness, not a fast/exact HF executor. Algebraic
identities are checked separately from native BF16 substitutions. No fallback,
future tokens, truth-table model compilation, or hardware timing is used.
"""
from __future__ import annotations
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
import argparse, base64, hashlib, json, math, platform, struct, sys
import numpy as np
import torch
import torch.nn.functional as NF

torch.set_num_threads(1)
torch.set_flush_denormal(False)
BF = torch.bfloat16

def bf(x):
    return torch.as_tensor(x, dtype=BF).clone()

def words(x):
    return x.detach().contiguous().view(torch.uint16).reshape(-1).tolist()

def native_mv(w, x):
    """Declared separate-FP32-product, balanced-add, final-BF16 ABI."""
    if w.ndim != 2 or x.ndim != 1 or w.shape[1] != x.numel():
        raise ValueError('incompatible projection shapes')
    if w.dtype != BF or x.dtype != BF:
        raise TypeError('BF16 operands required')
    p = w.float() * x.float().unsqueeze(0)
    width = 1 << (p.shape[1] - 1).bit_length()
    if width != p.shape[1]:
        p = NF.pad(p, (0, width - p.shape[1]))
    while p.shape[1] > 1:
        p = p[:, 0::2] + p[:, 1::2]
    return p[:, 0].to(BF)

def native_ffn(g, u, d, x):
    gx, ux = native_mv(g, x), native_mv(u, x)
    return native_mv(d, (NF.silu(gx) * ux).to(BF))

def parity_rewrite(g, u, d, x):
    """Candidate A: intentionally tested, NEVER claimed equivalent."""
    gx, ux = native_mv(g, x), native_mv(u, x)
    b = (gx * ux).to(BF)
    odd = (b * torch.tanh((gx.float() * .5)).to(BF)).to(BF)
    return ((native_mv(d, b).float() + native_mv(d, odd).float()) * .5).to(BF)

def qmatrix(a):
    return [[Q(float(v)) for v in row] for row in np.asarray(a)]

def compile_quadratic(g, u, d):
    """Explicit exact rational constructor of the real even component."""
    f, n, m = len(g), len(g[0]), len(d)
    if len(u) != f or any(len(v) != n for v in g+u) or any(len(v) != f for v in d):
        raise ValueError('invalid quadratic compiler shapes')
    pairs = [(j,k) for j in range(n) for k in range(j,n)]
    c = []
    for out in range(m):
        row=[]
        for j,k in pairs:
            row.append(sum((d[out][i] * (g[i][j]*u[i][k] if j==k else
                              g[i][j]*u[i][k]+g[i][k]*u[i][j]) for i in range(f)), Q(0)))
        c.append(row)
    return pairs,c

def eval_quadratic(pairs,c,x):
    return [sum((a*x[j]*x[k] for a,(j,k) in zip(row,pairs)),Q(0)) for row in c]

def direct_bilinear(g,u,d,x):
    gx=[sum((a*b for a,b in zip(row,x)),Q(0)) for row in g]
    ux=[sum((a*b for a,b in zip(row,x)),Q(0)) for row in u]
    return [sum((a*b*c for a,b,c in zip(row,gx,ux)),Q(0)) for row in d]

def compile_monomials(w):
    """Model-only dyadic-exponent program. No data-dependent oracle.

    Each column has a shared binary-power chain. Each output has explicit
    lists of powers for numerator and denominator. This re-encodes weights;
    it does not assume a smaller circuit can be discovered for free.
    """
    a=w.float().numpy(); rows,cols=a.shape
    fractions=qmatrix(a); shifts=[]; lengths=[]; exponents=[]
    for j in range(cols):
        shift=max(v[j].denominator.bit_length()-1 for v in fractions)
        shifts.append(shift)
    for row in fractions:
        exponents.append([int(v * (1<<shifts[j])) for j,v in enumerate(row)])
    for j in range(cols):
        lengths.append(max(1,max(abs(row[j]).bit_length() for row in exponents)))
    offsets=np.cumsum([0]+lengths[:-1]).tolist()
    pos=[]; neg=[]
    for row in exponents:
        pp=[];nn=[]
        for j,aij in enumerate(row):
            aa=abs(aij)
            for b in range(aa.bit_length()):
                if (aa>>b)&1:
                    (pp if aij>0 else nn).append(offsets[j]+b)
        pos.append(pp);neg.append(nn)
    refs=sum(map(len,pos))+sum(map(len,neg))
    power_count=sum(lengths)
    address_bytes=2 if power_count<=65536 else 4
    # Explicit serialized representation: i16 shifts, u16 lengths, u32 stream
    # offsets for each positive/negative row, and u16/u32 power references.
    packed_bytes=16 + 4*cols + 8*(rows+1) + address_bytes*refs
    mults=sum(max(0,l-1) for l in lengths)
    mults += sum(max(0,len(p)-1)+max(0,len(n)-1) for p,n in zip(pos,neg))
    return {'shifts':shifts,'lengths':lengths,'pos':pos,'neg':neg,
            'count':{'original_weight_bytes':2*rows*cols,'packed_program_bytes':packed_bytes,
                     'power_table_fp64_bytes':8*power_count,'power_references':refs,
                     'binary_power_and_product_multiplications':mults,
                     'divisions':sum(bool(n) for n in neg),
                     'exp_evaluations':cols,'log_evaluations':rows,
                     'reference_projection_multiplications':rows*cols,
                     'reference_projection_additions':rows*(cols-1)}}

def serialize_monomials(program):
    """Actually serialize the charged program, including its 16-byte header."""
    rows,cols=len(program['pos']),len(program['shifts'])
    addr=2 if sum(program['lengths'])<=65536 else 4
    payload=bytearray(struct.pack('<4sIII',b'VMP1',rows,cols,addr))
    payload.extend(struct.pack('<'+'h'*cols,*program['shifts']))
    payload.extend(struct.pack('<'+'H'*cols,*program['lengths']))
    flattened=[]
    for stream in [program['pos'],program['neg']]:
        offsets=[0];flat=[]
        for row in stream:
            flat.extend(row);offsets.append(len(flat))
        payload.extend(struct.pack('<'+'I'*(rows+1),*offsets))
        flattened.append(flat)
    for flat in flattened:
        payload.extend(struct.pack('<'+('H' if addr==2 else 'I')*len(flat),*flat))
    if len(payload)!=program['count']['packed_program_bytes']:
        raise AssertionError('packed layout accounting mismatch')
    return bytes(payload)

def deserialize_monomials(payload):
    if len(payload)<16:
        raise ValueError('truncated program')
    magic,rows,cols,addr=struct.unpack_from('<4sIII',payload,0)
    if magic!=b'VMP1' or addr not in [2,4] or rows==0 or cols==0:
        raise ValueError('invalid header')
    at=16
    def take(fmt,n):
        nonlocal at
        size=struct.calcsize('<'+fmt*n)
        if at+size>len(payload):raise ValueError('truncated program')
        result=list(struct.unpack_from('<'+fmt*n,payload,at));at+=size
        return result
    shifts=take('h',cols);lengths=take('H',cols)
    if any(l==0 for l in lengths):raise ValueError('empty power chain')
    offsets=[take('I',rows+1),take('I',rows+1)]
    streams=[]
    for off in offsets:
        if off[0]!=0 or any(a>b for a,b in zip(off,off[1:])):raise ValueError('invalid offsets')
        flat=take('H' if addr==2 else 'I',off[-1])
        if any(i>=sum(lengths) for i in flat):raise ValueError('invalid power reference')
        streams.append([flat[off[r]:off[r+1]] for r in range(rows)])
    if at!=len(payload):raise ValueError('unexpected trailing data')
    return {'shifts':shifts,'lengths':lengths,'pos':streams[0],'neg':streams[1]}

def monomial_projection(program,x):
    vals=[]
    for j,(shift,length) in enumerate(zip(program['shifts'],program['lengths'])):
        v=math.exp(math.ldexp(float(x[j]),-shift));vals.append(v)
        for _ in range(1,length):
            v=v*v; vals.append(v)
    out=[]
    for pp,nn in zip(program['pos'],program['neg']):
        numerator=math.prod(vals[i] for i in pp)
        denominator=math.prod(vals[i] for i in nn)
        e=numerator/denominator
        if not (e>0 and math.isfinite(e)):
            raise ArithmeticError('nonfinite monomial projection; no fallback')
        out.append(math.log(e))
    return np.array(out,dtype=np.float64)

def monomial_ffn(program,u,d,x,round_projection):
    gx=monomial_projection(program,x.float().numpy())
    if round_projection:
        gate=NF.silu(bf(gx))
    else:
        # Direct real transport: no BF16 projected gate checkpoint.
        gate=bf(gx/(1+np.exp(-gx)))
    return native_mv(d,(gate*native_mv(u,x)).to(BF))

def tensor_scale(n=16384,f=53248,m=16384):
    count=m*n*(n+1)//2; original=(2*f*n+m*f)
    return {'dimensions':{'input':n,'hidden':f,'output':m},
            'source':'Declared illustrative scale, not a loaded checkpoint',
            'original_projection_coefficients':original,
            'symmetric_quadratic_coefficients':count,
            'optimistic_two_bytes_per_tensor_coefficient_GiB':2*count/(1<<30),
            'original_BF16_projection_payload_GiB':2*original/(1<<30),
            'coefficient_ratio':count/original,
            'exact_tensor_coefficient_format':'arbitrary dyadic rational; two-byte assumption above is optimistic, NOT exact encoding',
            'dense_constructor_scalar_multiply_upper_bound':3*m*f*n*n,
            'tensor_query_scalar_multiply_count':n*(n+1)//2+count,
            'odd_direct_projection_coefficient_reads':original,
            'odd_direct_branch_alone_vs_original_projection_reads':1.0}

def attention_witness():
    wk=bf([[1,1],[1,2]]); hist=bf([[0,0],[1,256]]);q=bf([2,-1])
    ks=torch.stack([native_mv(wk,h) for h in hist])
    native_scores=native_mv(ks,q)
    pulled=native_mv(wk.T.contiguous(),q)
    transported_scores=native_mv(hist,pulled)
    a0=torch.softmax(native_scores.float()/math.sqrt(2),dim=0).to(BF)
    a1=torch.softmax(transported_scores.float()/math.sqrt(2),dim=0).to(BF)
    # Values [0,1] make the actual attention result expose the probability.
    v=bf([[0,1]])
    o0=native_mv(v,a0);o1=native_mv(v,a1)
    return {'Wk':wk.float().tolist(),'history':hist.float().tolist(),'query':q.float().tolist(),
            'projected_keys':ks.float().tolist(),'pulled_query':pulled.float().tolist(),
            'native_scores':native_scores.float().tolist(),
            'transported_scores':transported_scores.float().tolist(),
            'native_output':o0.float().tolist(),'transported_output':o1.float().tolist(),
            'native_words':words(o0),'transported_words':words(o1),
            'scope':'One explicit rounded attention operation, not a full HF decoder or RNG/state test'}

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    reg=json.loads((root/'PREREGISTRATION.json').read_text())
    rows=[]; counts=[]; serialized=[]; max_real_error=0.0
    for seed in reg['seeds']:
        rng=np.random.default_rng(seed);n=8;f=16;m=8
        g=bf(rng.normal(size=(f,n))/math.sqrt(n));u=bf(rng.normal(size=(f,n))/math.sqrt(n));d=bf(rng.normal(size=(m,f))/math.sqrt(f))
        program=compile_monomials(g);data=serialize_monomials(program)
        counts.append({'seed':seed,**program['count']})
        serialized.append({'seed':seed,'format':'VMP1','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()})
        decoded=deserialize_monomials(data)
        assert all(decoded[k]==program[k] for k in decoded)
        program=decoded
        for query in range(reg['inputs_per_seed']):
            x=bf(rng.normal(size=n));native=native_ffn(g,u,d,x)
            candidate=parity_rewrite(g,u,d,x)
            exactg=g.double()@x.double();exactu=u.double()@x.double()
            real=d.double()@(NF.silu(exactg)*exactu)
            real_split=.5*d.double()@(exactg*exactu)+.5*d.double()@(exactg*exactu*torch.tanh(exactg/2))
            err=float((real-real_split).abs().max());max_real_error=max(max_real_error,err)
            rec={'seed':seed,'query':query,'output_words':m,'input_words':words(x),
                 'native_output_words':words(native),'parity_rewrite_words':words(candidate),
                 'parity_mismatches':sum(a!=b for a,b in zip(words(native),words(candidate))),
                 'float64_identity_max_absolute_error':err}
            for mode in [False,True]:
                key='monomial_gate_rounded' if mode else 'monomial_direct'
                try:
                    y=monomial_ffn(program,u,d,x,mode)
                    rec[key+'_words']=words(y)
                    rec[key+'_mismatches']=sum(a!=b for a,b in zip(words(native),words(y)))
                except (OverflowError,ArithmeticError,ValueError) as e:
                    rec[key+'_error']=str(e)
            rows.append(rec)
    # Fully dense, full-column-rank G and U, exact zero quadratic tensor.
    gq=qmatrix([[1,1],[1,-1],[1,2],[1,-2]])
    uq=[[a*s for a in row] for row,s in zip(gq,[2,4,8,16])]
    dq=qmatrix([[-1,.5,.125,-.0625]])
    pairs,coeffs=compile_quadratic(gq,uq,dq)
    qzero=all(c==0 for row in coeffs for c in row)
    gt=bf([[float(v) for v in row] for row in gq]);ut=bf([[float(v) for v in row] for row in uq]);dt=bf([[float(v) for v in row] for row in dq])
    x=bf([1,2]); y=native_ffn(gt,ut,dt,x);ym=native_ffn(gt,ut,dt,-x)
    zero_example={'G':gt.float().tolist(),'U':ut.float().tolist(),'D':dt.float().tolist(),
                  'x':x.float().tolist(),'quadratic_coefficients':[[str(v) for v in row] for row in coeffs],
                  'exact_zero_quadratic':qzero,'native_output':y.float().tolist(),
                  'native_minus_input_output':ym.float().tolist(),'native_words':words(y),
                  'all_weight_entries_nonzero':all(v!=0 for mat in [gq,uq,dq] for row in mat for v in row),
                  'scope':'Explicit synthetic standard SwiGLU block; refutes retaining only its quadratic even summary'}
    # Exact compiler/property check, fixed small polynomial (no model truth table).
    eg=qmatrix([[1,2,3],[-1,1,2],[2,-1,1],[1,1,-2]])
    eu=qmatrix([[2,1,1],[1,3,1],[-1,1,2],[3,1,2]])
    ed=qmatrix([[1,-2,3,1],[-1,1,2,-3]])
    ep,ec=compile_quadratic(eg,eu,ed);checks=0
    for xx in product(reg['exact_input_grid'],repeat=3):
        xx=list(map(Q,xx));assert eval_quadratic(ep,ec,xx)==direct_bilinear(eg,eu,ed,xx);checks+=2
    # Native monomial transport counterexample with no nonfinite values.
    gg=bf([[1,1/256]]);uu=bf([[.5,.5]]);dd=bf([[1]]);xx=bf([1,1])
    pp=compile_monomials(gg)
    direct=native_ffn(gg,uu,dd,xx);transport=monomial_ffn(pp,uu,dd,xx,False)
    transport_example={'G':gg.float().tolist(),'U':uu.float().tolist(),'D':dd.float().tolist(),
      'x':xx.float().tolist(),'native_gate':native_mv(gg,xx).float().tolist(),
      'real_gate_exact':'257/256','native_output':direct.float().tolist(),
      'transported_output':transport.float().tolist(),
      'native_words':words(direct),'transported_words':words(transport)}
    # Even exact real reconstruction is not an original FP32 reduction.
    ww=bf([[2**24,1,-2**24,1]]);xx4=bf([1,1,1,1])
    accumulation={'W':ww.float().tolist(),'x':xx4.float().tolist(),
                  'native_balanced_FP32_then_BF16':native_mv(ww,xx4).float().tolist(),
                  'exact_rational_sum':str(sum((Q(float(v)) for v in ww.flatten()),Q(0)))}
    summary={
      'scope':'E0/E1 scoped construction and cheapest rejection gates; no new mission core',
      'preregistration_sha256':hashlib.sha256((root/'PREREGISTRATION.json').read_bytes()).hexdigest(),
      'seed_count':len(reg['seeds']),'full_block_inputs':len(rows),
      'compared_output_words':sum(r['output_words'] for r in rows),
      'native_parity_rewrite_mismatches':sum(r['parity_mismatches'] for r in rows),
      'monomial_direct_mismatches':sum(r.get('monomial_direct_mismatches',0) for r in rows),
      'monomial_direct_errors':sum('monomial_direct_error' in r for r in rows),
      'monomial_gate_rounded_mismatches':sum(r.get('monomial_gate_rounded_mismatches',0) for r in rows),
      'monomial_gate_rounded_errors':sum('monomial_gate_rounded_error' in r for r in rows),
      'float64_identity_max_absolute_error':max_real_error,
      'exact_rational_quadratic_outputs_checked':checks,
      'zero_quadratic_nonzero_output':zero_example,
      'native_monomial_counterexample':transport_example,
      'native_accumulation_counterexample':accumulation,
      'attention_transport_counterexample':attention_witness(),
      'scale':tensor_scale(),
      'monomial_programs':counts,
      'mission':{'THEORY_STATUS':'NOT_ESTABLISHED','CORE_ADMISSION':False,'O1_O6':'OPEN',
                 'HARDWARE_STATUS':'NOT_TESTED','public_checkpoint_tested':False,
                 'three_qualifying_new_principles_constructed':False},
      'performance':'All counts are logical/derived; no native4B baseline, GPU VRAM, TTFT or latency measurement',
      'decision':'Do not enlarge these three unqualified routes. No fast residual/word-corrector has been constructed.'}
    (out/'serialized_programs.json').write_text(json.dumps(serialized,sort_keys=True,indent=2)+'\n')
    (out/'raw.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    env={'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'platform':platform.platform(),'cuda_available':torch.cuda.is_available(),'torch_threads':torch.get_num_threads(),'ABI':reg['abi']}
    (out/'environment.json').write_text(json.dumps(env,sort_keys=True,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
