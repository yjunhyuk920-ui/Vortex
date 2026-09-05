"""Deterministic local reference audit; no downloaded model or GPU is involved."""
from __future__ import annotations
import base64, hashlib, itertools, json, random, struct, sys
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from reference import BFloat, PrefixSource, generate, oracle, operand_model, p2, rounded, stored, trunc, floor_log2, prefix_interval, interval_cell

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/native_observable'
OUT.mkdir(parents=True, exist_ok=True)

def fp32_bits(q):
    return struct.unpack('<I', struct.pack('<f', float(q)))[0]

def bf_bits(q):
    return fp32_bits(q) >> 16

def val32(u):
    return struct.unpack('<f', struct.pack('<I', u))[0]

def rng_word(rng):
    return BFloat(rng.choice([-1,1]), rng.randint(-3,3), rng.randrange(128))

def audit():
    cfg = json.loads((ROOT/'experiments/native_observable/preregistration.json').read_text())
    rng = random.Random(cfg['controls']['seed'])
    captures, packed, input_digest = [], bytearray(), hashlib.sha256()
    mismatches = []
    for i in range(cfg['controls']['random_blocks']):
        w, x = [rng_word(rng) for _ in range(8)], [rng_word(rng) for _ in range(8)]
        input_digest.update(json.dumps([[v.sign,v.exponent,v.mantissa] for v in w+x],separators=(',',':')).encode())
        result = generate(PrefixSource(w), x)
        expected = oracle(w,x)
        if result['value'] != expected:
            mismatches.append(i)
        row = dict(index=i, output_bf16=bf_bits(result['value']),
                   lo_fp32=fp32_bits(result['lo']), hi_fp32=fp32_bits(result['hi']),
                   known=result['known'])
        captures.append(row)
        mask = sum(k << (3*j) for j,k in enumerate(result['known']))
        packed.extend(struct.pack('<HII',row['output_bf16'],row['lo_fp32'],row['hi_fp32']))
        packed.extend(mask.to_bytes(3,'little'))
    assert not mismatches

    # A bit-exact pre-store and post-store witness. Both operand pairs have
    # identical mathematical products. This is a MODEL prediction, not a GPU run.
    a = [F(3,2)] + [p2(-12)]*4 + [F(0)]*3
    b = list(a)
    a2, b2 = list(a), list(b)
    a2[0],b2[0] = F(1),F(9,4)
    u,v = operand_model(a,b,p2(-7)),operand_model(a2,b2,p2(-7))
    assert [aa*bb for aa,bb in zip(a,b)] == [aa*bb for aa,bb in zip(a2,b2)]
    assert stored(u) == F(145,64) and stored(v) == F(9,4)
    # Same store is possible despite different private FP32 results.
    private_a,private_b=F(1),F(1)+p2(-23)
    assert private_a != private_b and stored(private_a) == stored(private_b)

    # Exhaust all 4^8 completions of a fixed, predeclared 5-bit prefix box.
    exp = [-3,-2,-1,0,1,2,3,0]
    bases = [12,28,44,60,76,92,108,124]
    x = [BFloat((-1)**j, exp[7-j], (23+17*j)%128) for j in range(8)]
    headers = [((-1)**(j//2),exp[j]) for j in range(8)]
    E = max(headers[j][1]+x[j].exponent for j in range(8))
    unit=p2(E-24)
    pairs=[]
    for j in range(8):
        lo,hi=prefix_interval(headers[j],bases[j]>>2,5)
        values=[lo*x[j].value(),hi*x[j].value()]
        pairs.append((trunc(min(values)/unit),trunc(max(values)/unit)))
    lo=rounded(sum(p[0] for p in pairs)*unit,24,False)
    hi=rounded(sum(p[1] for p in pairs)*unit,24,False)
    checked=0
    for suffix in itertools.product(range(4),repeat=8):
        w=[BFloat(headers[j][0],headers[j][1],bases[j]+suffix[j]) for j in range(8)]
        y=oracle(w,x)
        assert stored(lo) <= y <= stored(hi)
        checked+=1

    # Independent bit-level BF16 rounding of already representable FP32 values,
    # and host nextafter control for FP32 toward-zero. No GPU/native-TC claim.
    nr=0
    for _ in range(cfg['controls']['rounding_cases']):
        q=F(rng.randint(-(1<<25),(1<<25)),1<<20)
        rz=rounded(q,24,False)
        f=np.float32(float(q))
        if abs(float(f)) > abs(float(q)):
            f=np.nextafter(f,np.float32(0))
        assert float(rz)==float(f)
        raw=fp32_bits(rz)
        target=(raw+0x7fff+((raw>>16)&1))>>16
        assert bf_bits(stored(rz))==target
        nr+=1
    assert interval_cell(F(1)+p2(-8),F(1)+p2(-8)+p2(-24)) is None
    assert interval_cell(-F(1)-p2(-8)-p2(-24),-F(1)-p2(-8)) is None
    assert interval_cell(F(-1),F(1)) is None
    assert interval_cell(F(0),F(0)) == 0

    # Replay the binary capture to the exact same canonical JSONL.
    restored=[]
    for i in range(len(captures)):
        out,lb,hb=struct.unpack('<HII',packed[13*i:13*i+10])
        mask=int.from_bytes(packed[13*i+10:13*i+13],'little')
        restored.append(dict(index=i,output_bf16=out,lo_fp32=lb,hi_fp32=hb,known=[(mask>>(3*j))&7 for j in range(8)]))
    canonical=lambda rows: ''.join(json.dumps(z,sort_keys=True,separators=(',',':'))+'\n' for z in rows).encode()
    raw=canonical(captures)
    assert canonical(restored)==raw
    (OUT/'cases.jsonl').write_bytes(raw)
    sums=[sum(z['known']) for z in captures]
    summary={
        'scope':'Declared normal BF16 eight-product block and actual final BF16 store; no real GPU or whole-kernel validation',
        'THEORY_STATUS':'NOT_ESTABLISHED','HARDWARE_STATUS':'NOT_TESTED','CORE_ADMISSION':False,
        'random_blocks':len(captures),'mismatches':mismatches,
        'weight_input_sha256':input_digest.hexdigest(),
        'prefix_bits_min':min(sums),'prefix_bits_max':max(sums),'prefix_bits_mean':sum(sums)/len(sums),
        'logical_weight_bits_mean':72+sum(sums)/len(sums),'native_weight_bits':128,
        'endpoint_products_mean':16+2*sum(sums)/len(sums),
        'early_certificate_blocks':sum(s<56 for s in sums),
        'all_headers_traffic_floor_fraction':9/16,
        'header_only_maximum_logical_reduction':16/9,
        'exhaustive_completions':checked,'rounding_controls':nr,
        'operand_witness':{'native_model_A':str(u),'native_model_B':str(v),'stored_A':str(stored(u)),'stored_B':str(stored(v)), 'stored_A_bits':hex(bf_bits(stored(u))), 'stored_B_bits':hex(bf_bits(stored(v)))},
        'private_difference_erased_at_store':True,
        'capture_encoding':'little endian per case: uint16 stored BF16, uint32 lower FP32, uint32 upper FP32, uint24 eight 3-bit known-prefix lengths. Input generation is pinned in audit.py/preregistration; not a compressed model.',
        'capture_b64':base64.b64encode(packed).decode(),
        'decoded_cases_sha256':hashlib.sha256(raw).hexdigest(),
        'not_run':cfg['unexecuted'],
        'limitations':['No GPU timing or hardware fidelity established','No extra BF16 cut may be inserted inside native FP32 accumulation','Signed zero/NaN/Inf/subnormal hardware behavior not modeled','No general cheap source or target-scale cost closure']
    }
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    return {k:v for k,v in summary.items() if k!='capture_b64'}

if __name__=='__main__':
    print(json.dumps(audit(),ensure_ascii=False,indent=2))
