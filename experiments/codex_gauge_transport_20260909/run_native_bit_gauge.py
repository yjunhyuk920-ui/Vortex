import argparse
import hashlib
import json
from pathlib import Path
import random

from native_bit_gauge import bf16_multiply, butterfly, cost, encoded_multiply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("output directory must not exist")
    args.output_dir.mkdir(parents=True)
    rng_before = random.getstate()
    for word in range(65536):
        row = [word,word ^ 0xffff,((word << 1) | (word >> 15)) & 0xffff,0x8000]
        assert butterfly(butterfly(row),inverse=True) == row
    golden = [(0x3f80,0x4000,0x4000),(0xbf80,0x4000,0xc000),
              (0x0000,0xbf80,0x8000),(0x8000,0xbf80,0x0000),
              (0x0001,0x3f80,0x0001),(0x7f80,0x0000,0x7fc0),
              (0x7f80,0x3f80,0x7f80),(0x7fc1,0x3f80,0x7fc0)]
    for a,b,expected in golden:
        assert bf16_multiply(a,b) == expected
    alphabet = [0x0000,0x8000,0x0001,0x007f,0x3f80,0xbf80,0x4000,0x3f81,0x7f7f,0x7f80,0xff80,0x7fc1]
    records = []
    for i,a in enumerate(alphabet):
        for j,b in enumerate(alphabet):
            x = [a,alphabet[(i+1)%12],a ^ 0x8000,alphabet[(i+5)%12]]
            y = [b,alphabet[(j+3)%12],b ^ 0x8000,alphabet[(j+7)%12]]
            expected = [bf16_multiply(u,v) for u,v in zip(x,y)]
            got = butterfly(encoded_multiply(butterfly(x),butterfly(y)),inverse=True)
            assert got == expected
            records.append({"x":x,"y":y,"output":got})
    state = [0x3f80,0xbf80,0x0001,0x8000]
    encoded_state = butterfly(state)
    chain = []
    for t in range(20):
        incoming = [alphabet[(3*t+j)%12] for j in range(4)]
        state = [bf16_multiply(a,b) for a,b in zip(state,incoming)]
        encoded_state = encoded_multiply(encoded_state,butterfly(incoming))
        assert butterfly(encoded_state,inverse=True) == state
        chain.append({"step":t,"input":incoming,"native_state":state,"encoded_state":encoded_state})
    assert random.getstate() == rng_before
    summary = {
        "kind":"EXPLICIT_NATIVE_BIT_GAUGE_HADAMARD_CONSTRUCTOR",
        "roundtrip_vectors":65536,"roundtrip_word_coordinates":4*65536,
        "product_vectors":len(records),"product_word_coordinates":4*len(records),
        "known_product_words":len(golden),"state_chain_steps":20,
        "mismatches":0,"python_rng_state_unchanged":True,
        "abi":"BF16 operands / separate FP32 RNE product / BF16 RNE store / canonical quiet NaN",
        "cost_at_16384":cost(16384),"new_qualifying_principles":0,
        "full_HF_state_RNG_KV_tested":False,"target_hardware_tested":False,
        "theory_status":"NOT_ESTABLISHED","hardware_status":"NOT_TESTED","core_admission":False,
        "claim_boundary":"Numeric-field monomial lemma does not exclude cheap opaque-bit encodings; arbitrary native dense projection is still missing.",
    }
    for name,value in (("summary.json",summary),("native_word_cases.json",records),("state_chain.json",chain)):
        (args.output_dir/name).write_text(json.dumps(value,sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\r\n")
    paths = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    (args.output_dir/"checksums.sha256").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in paths),encoding="utf-8",newline="\r\n")
    print(json.dumps(summary,sort_keys=True,indent=2))


if __name__ == "__main__":
    main()
