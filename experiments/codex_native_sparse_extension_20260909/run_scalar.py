"""Frozen deterministic controls for scalar_producer.py; writes immutable v2 evidence."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from scalar_producer import compile_store, project, project_fp32, full_reference, full_reference_fp32, cost_formula, _f32_to_bf16_word

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "scalar_results_v2"

def bits_to_float_word(sign, exponent, fraction): return (sign << 15) | (exponent << 7) | fraction
def fp32_bits(value):
    import struct
    return struct.unpack(">I", struct.pack(">f", value))[0]

def check_case(matrix, inputs, signs, label):
    store = compile_store(matrix, signs)
    sparse_fp, stats = project_fp32(store, inputs)
    full_fp = full_reference_fp32(matrix, inputs)
    sparse = [_f32_to_bf16_word(x) for x in sparse_fp]
    full = [_f32_to_bf16_word(x) for x in full_fp]
    if [fp32_bits(x) for x in sparse_fp] != [fp32_bits(x) for x in full_fp] or sparse != full:
        raise AssertionError(label)
    return stats

def main():
    checks = 0
    # Exactly 2 * 255 * 128 = 65,280 finite BF16 encodings.
    finite = [bits_to_float_word(s,e,f) for s in range(2) for e in range(255) for f in range(128)]
    assert len(finite) == 65280
    dyadic = [0x3F80,0xBF80,0x4000,0xC000,0x3F00,0xBF00]
    # Selected coordinate 1; width 3 and width 4, both inactive-zero patterns.
    for word in finite:
        for x in dyadic:
            for signs in ((0,0,0),(1,0,1)):
                inputs = [(signs[0]<<15), x, (signs[2]<<15)]
                check_case([[0x0000,word,0x8000],[0x8000,word,0x0000]], inputs, list(signs), "enumeration-w3")
                checks += 1
            for signs in ((0,1,0,1),(1,0,1,0)):
                inputs = [(signs[0]<<15), x, (signs[2]<<15), (signs[3]<<15)]
                check_case([[0x0000,word,0x8000,0x0000]], inputs, list(signs), "enumeration-w4")
                checks += 1
    fixed = [
        ([[0x3F80,0x4040,0x3E80],[0x8000,0xBF80,0x3F80]],[0x3F80,0x3E80,0x8000],[0,0,1]),
        ([[0x7F7F,0x7F7F,0x0080,0x8080],[0x8001,0x0001,0x3F80,0xBF80]],[0x4000,0x4000,0x8000,0x0000],[0,0,1,0]),
        ([[0x0001,0x8001,0x3F80],[0x8000,0x0000,0xBF80]],[0x3F00,0xBF00,0x8000],[0,0,1]),
        ([[0x3F80,0x3F80,0x8000,0x0000]],[0x3F80,0xBF80,0x8000,0x0000],[0,0,1,0]),
    ]
    for matrix, inputs, signs in fixed:
        check_case(matrix, inputs, signs, "fixed")
        checks += 1
    negatives = 0
    store = compile_store([[0x3F80,0x3F80,0x3F80]], [0,0,0])
    for bad in ([0x7F80,0,0], [0x3F80,0x3F80,0x3F80], [0x8000,0,0]):
        try: project(store,bad)
        except ValueError: negatives += 1
    try: compile_store([[0x7F80]], [0])
    except ValueError: negatives += 1
    # Red regression retained: the old max_support=3 route silently summed
    # only two leaves (2) although the full reference is 3.  v2 must refuse.
    regression_store = compile_store([[0x3F80,0x3F80,0x3F80]], [0,0,0])
    assert full_reference([[0x3F80,0x3F80,0x3F80]], [0x3F80]*3) == [0x4040]
    try: project(regression_store, [0x3F80,0x3F80,0x3F80], max_support=3)
    except ValueError: negatives += 1
    assert negatives == 5
    OUT.mkdir(exist_ok=False)
    report = {"finite_bf16_count":len(finite),"enumerated_cases":checks,"negative_rejections":negatives,
              "regression_max_support_3":"refused; legacy dropped-third result=2 while full reference=3",
              "cost_m16384_n16384_k1":cost_formula(k=1),"cost_m16384_n16384_k2":cost_formula(k=2),
              "cost_m16384_n16384_kn":cost_formula(k=16384),"random_search":"none"}
    (OUT/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\n")
    digest=hashlib.sha256((OUT/"report.json").read_bytes()).hexdigest()
    (OUT/"SHA256SUMS").write_text(f"{digest}  report.json\n",encoding="ascii",newline="\n")
    print(json.dumps(report,sort_keys=True))
if __name__ == "__main__": main()
