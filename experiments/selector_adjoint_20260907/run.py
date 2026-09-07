#!/usr/bin/env python3
"""Deterministic auxiliary algebra/cost gate, not a neural inference engine."""
from pathlib import Path
import sys,json,hashlib,subprocess,random
from fractions import Fraction
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'src'))
from adjoint import Program,Builder,native_rounder,transpose_binary_matrix
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'run'

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def planes(words,width):
    return [int.from_bytes(np.packbits(((words>>i)&1).astype(np.uint8),bitorder='little').tobytes(),'little') for i in range(width)]
def unpack(p,lanes,width=32):
    out=np.zeros(lanes,dtype=np.uint32)
    for i,q in enumerate(p[:width]):
        bits=np.unpackbits(np.frombuffer(q.to_bytes((lanes+7)//8,'little'),dtype=np.uint8),bitorder='little')[:lanes]
        out|=bits.astype(np.uint32)<<i
    return out

def exact_value(word,width=32):
    fracbits=23 if width==32 else 7
    e=(word>>fracbits)&255;f=word&((1<<fracbits)-1)
    if e==255:raise ValueError('nonfinite')
    m=f if not e else f+(1<<fracbits)
    shift=(e if e else 1)-127-fracbits
    return Fraction(m<<shift) if shift>=0 else Fraction(m,1<<(-shift))
def fraction_round(word):
    sign=(word>>16)&0x8000;e=(word>>23)&255;f=word&0x7fffff
    if e==255:return 0x7fc0 if f else sign|0x7f80
    x=exact_value(word&0x7fffffff)
    if x>=Fraction(511)*(1<<119):return sign|0x7f80
    lo,hi=0,0x7f7f
    while lo<hi:
        mid=(lo+hi+1)//2
        if exact_value(mid,16)<=x:lo=mid
        else:hi=mid-1
    if lo==0x7f7f:return sign|lo
    low,high=exact_value(lo,16),exact_value(lo+1,16)
    if x-low>high-x or (x-low==high-x and lo&1):lo+=1
    return sign|lo

def native_run():
    out=OUT/'native';out.mkdir(parents=True,exist_ok=True)
    hi=np.repeat(np.arange(65536,dtype=np.uint32),5)<<16
    low=np.tile(np.array([0,32767,32768,32769,65535],dtype=np.uint32),65536)
    rng=np.random.default_rng(17029)
    words=np.concatenate([hi|low,rng.integers(0,1<<32,4096,dtype=np.uint32)])
    raw=words.astype('<u4').tobytes();(out/'inputs.u32').write_bytes(raw)
    p,roots=native_rounder();(out/'program.sla').write_bytes(p.to_bytes())
    p=Program.from_bytes((out/'program.sla').read_bytes())
    got=p.extract(planes(words,32),len(words))
    outputs=unpack(got[:16],len(words),16).astype('<u2')
    state=unpack(got[16:],len(words),32).astype('<u4')
    (out/'outputs.u16').write_bytes(outputs.tobytes());(out/'state.u32').write_bytes(state.tobytes())
    build=ROOT/'build';build.mkdir(exist_ok=True)
    cmd=['cc','-std=c11','-O2','-fno-fast-math','-ffp-contract=off',str(ROOT/'src'/'reference.c'),'-lm','-o',str(build/'reference')]
    subprocess.run(cmd,check=True,capture_output=True)
    res=subprocess.run([str(build/'reference')],input=raw,capture_output=True,check=True)
    if len(res.stdout)!=2*len(words):raise RuntimeError('C output count')
    (out/'reference.u16').write_bytes(res.stdout)
    reference=np.frombuffer(res.stdout,dtype='<u2')
    mismatches=int(np.count_nonzero(outputs!=reference));state_errors=int(np.count_nonzero(state!=words))
    fractions=[]
    for j in np.linspace(0,len(words)-1,512,dtype=int):
        expected=fraction_round(int(words[j]))
        fractions.append(dict(input=int(words[j]),actual=int(outputs[j]),expected=expected))
    assert mismatches==state_errors==0
    assert all(s['actual']==s['expected'] for s in fractions)
    dump(out/'fraction_checks.json',fractions)
    # This checks conversion primitive only, not any pretrained neural layer.
    return dict(inputs=len(words),output_words=len(words),preserved_state_words=len(words),
                output_mismatches=mismatches,state_mismatches=state_errors,
                fraction_checks=len(fractions),cost=p.costs(),
                scope='FP32 word->BF16 conversion + raw input-word liveness. Canonical NaN ABI. No model weights or neural acceleration.',
                direct_source_coefficient_gates=p.costs()['coefficient_bit_gates'])

def matrix_run():
    rows=[]
    for n in (16,32,64,128):
        for seed in (17,29):
            d=OUT/f'gf2_{n}_{seed}';d.mkdir(parents=True,exist_ok=True)
            rng=np.random.default_rng(seed)
            w=rng.integers(0,2,(n,n),dtype=np.uint8)
            x=rng.integers(0,2,(256,n),dtype=np.uint8)
            (d/'weights.bits').write_bytes(np.packbits(w.flatten(),bitorder='little').tobytes())
            (d/'inputs.bits').write_bytes(np.packbits(x.flatten(),bitorder='little').tobytes())
            p=transpose_binary_matrix(w.tolist());(d/'program.sla').write_bytes(p.to_bytes())
            p=Program.from_bytes((d/'program.sla').read_bytes())
            ip=[int.from_bytes(np.packbits(x[:,j],bitorder='little').tobytes(),'little') for j in range(n)]
            yp=p.extract(ip,256)
            y=np.stack([np.unpackbits(np.frombuffer(z.to_bytes(32,'little'),dtype=np.uint8),bitorder='little') for z in yp],axis=1)
            ref=((x.astype(np.int64)@w.astype(np.int64).T)&1).astype(np.uint8)
            errs=int(np.count_nonzero(y!=ref));assert not errs
            (d/'outputs.bits').write_bytes(np.packbits(y.flatten(),bitorder='little').tobytes())
            (d/'reference.bits').write_bytes(np.packbits(ref.flatten(),bitorder='little').tobytes())
            cost=p.costs();cost.update(packed_baseline_bytes=n*n//8,ones=int(w.sum()),
                                      program_to_packed_ratio=len(p.to_bytes())/(n*n//8))
            rows.append(dict(n=n,seed=seed,queries=256,output_bits=256*n,mismatches=errs,cost=cost))
    return rows

def random_run():
    rec=[]
    words=np.arange(256,dtype=np.uint32);ip=planes(words,8)
    for seed in range(100,112):
        r=random.Random(seed);g=Builder(8,16);ids=[g.input(j) for j in range(8)]
        for k in range(64):
            a,b=r.choice(ids),r.choice(ids)
            ids.append(g.and_(a,b) if r.randrange(2) else g.xor(a,b))
        roots=[r.choice(ids[-32:]) for _ in range(16)];p=g.wrap(roots)
        d=OUT/f'boolean_{seed}';d.mkdir(exist_ok=True)
        (d/'program.sla').write_bytes(p.to_bytes());dump(d/'roots.json',roots)
        p=Program.from_bytes((d/'program.sla').read_bytes())
        ref=[p.coefficients(ip,256)[i] for i in roots]
        actual=p.extract(ip,256);assert actual==ref
        probes=[]
        for j in range(16):
            sel=[0]*16;sel[j]=(1<<256)-1;probes.append(p.scalar(ip,sel,256))
        assert actual==probes
        dump(d/'output_bitplanes.json',[format(v,'064x') for v in actual])
        rec.append(dict(seed=seed,inputs=256,output_bits=4096,mismatches=0,cost=p.costs()))
    return rec

def rational_rank(a):
    a=[[Fraction(x) for x in row] for row in a];r=0
    for j in range(len(a[0])):
        pivot=next((i for i in range(r,len(a)) if a[i][j]),None)
        if pivot is None:continue
        a[r],a[pivot]=a[pivot],a[r];v=a[r][j];a[r]=[x/v for x in a[r]]
        for i in range(r+1,len(a)):
            v=a[i][j]
            if v:a[i]=[x-v*y for x,y in zip(a[i],a[r])]
        r+=1
        if r==len(a):break
    return r

def scope_run():
    w=np.array([2**24,1,-2**24,1],dtype=np.float32)
    balanced=np.float32(np.float32(w[0]+w[1])+np.float32(w[2]+w[3]))
    rev=np.float32(0)
    for v in w[::-1]:rev=np.float32(rev+v)
    assert float(balanced)==1 and float(rev)==2
    # Type trap: z and z^2 agree on Boolean values, formal derivative at zero differs.
    square=dict(z=[0,1],z_values=[0,1],z_squared_values=[0,1],formal_d_z_at_zero=1,formal_d_z2_at_zero=0,
                consequence='Never Boolean-simplify selector-bearing products; typed source forbids them.')
    energy=[]
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                E=(a-1)**2+(b-a)**2+(c-a)**2
                energy.append(dict(state=[a,b,c],energy=E))
    local=(1,2,2,2) # E(000), E(100), E(010), E(001)
    assert sum(t['energy']==0 for t in energy)==1
    h=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],dtype=np.int64)
    frames=[dict(directions=k,rank=rational_rank((h[:k].T@h[:k]).tolist())) for k in range(1,5)]
    assert (h.T@h==4*np.eye(4,dtype=np.int64)).all()
    return dict(native_transpose=dict(weights=[int(v) for v in w],inputs=[1,1,1,1],balanced=float(balanced),reverse_accumulation=float(rev)),
                selector_type_trap=square,energy_assignments=energy,
                energy_local_minimum=dict(state=[0,0,0],energy=1,one_flip_energies=[2,2,2],unique_solution=[1,1,1]),
                directional=dict(target=[1,0],one_direction=[1,1],one_sample_reconstruction=[1,1],
                                 exact_four_direction_average=[1,0],frame4=h.tolist(),frame_ranks=frames,
                                 scope='Fixed input-independent linear directional reconstruction only; not all encodings or nonlinear decoders.'))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    native=native_run();mat=matrix_run();randoms=random_run();scope=scope_run()
    dump(OUT/'scope.json',scope)
    summary=dict(native_conversion=native,gf2_matrices=mat,random_boolean_circuits=randoms,
                 status=dict(THEORY_STATUS='NOT_ESTABLISHED',AUXILIARY='SELECTOR_LINEAR_EXTRACTION_VERIFIED_IN_MODEL',
                             CORE_ADMISSION=False,THREE_QUALIFYING_NEW_PRINCIPLES=False,HARDWARE_STATUS='NOT_TESTED'),
                 native_model_speedup_measured=False,
                 gate='No cheap whole-native scalar source was constructed. Do not expand a neural backend from an algebraic extraction theorem.')
    dump(OUT/'summary.json',summary)
    files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    dump(OUT/'manifest.json',files)
    print(json.dumps(dict(files=len(files),native_words=native['inputs'],gf2_output_bits=sum(r['output_bits'] for r in mat),
                          boolean_output_bits=sum(r['output_bits'] for r in randoms),native_cost=native['cost']),indent=2))
if __name__=='__main__':main()
