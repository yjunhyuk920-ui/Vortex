#!/usr/bin/env python3
"""Exact finite-word certificate; no floating-point or model/backend execution."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def value(word):
    assert 0 <= word < 65536
    sign = -1 if word & 0x8000 else 1
    exponent = (word >> 7) & 255
    fraction = word & 127
    assert exponent != 255, "finite input only"
    if exponent == 0:
        significand, shift = fraction, -133
    else:
        significand, shift = 128 + fraction, exponent - 134
    return sign * Fraction(significand) * Fraction(2) ** shift


def rank(vectors):
    pivots = {}
    for v in vectors:
        while v:
            p = v.bit_length() - 1
            if p in pivots:
                v ^= pivots[p]
            else:
                pivots[p] = v
                break
    return len(pivots)


def main():
    graph_path = ROOT / "experiments/native_global_transition_20260908/results/probe_graph/graph_0.json"
    graph = json.loads(graph_path.read_text())
    nodes = {n["name"]: n for n in graph["nodes"]}
    assert nodes["silu"]["target"] == "aten.silu.default"
    assert nodes["mul_12"]["target"] == "aten.mul.Tensor"
    assert nodes["mul_12"]["args"] == {"tuple": [{"node": "silu"}, {"node": "_unsafe_view_5"}]}
    for key in ("_param_constant7", "_param_constant8", "_param_constant9"):
        assert graph["attributes"][key]["meta"]["dtype"] == "torch.bfloat16"
    records = []
    for j in range(15):
        b0, b1 = 0x4000, 0x4000 ^ (1 << j)
        # The quartet is an affine plane in the 32 original operand bits.
        inputs = [(0, b0), (0x3f80, b0), (0, b1), (0x3f80, b1)]
        outputs = [0, b0, 0, b1]
        input_xor = output_xor = 0
        products = []
        for (a, b), out in zip(inputs, outputs):
            assert not (a & 0x8000 or b & 0x8000 or out & 0x8000)
            for word in (a, b, out):
                assert word == 0 or 1 <= ((word >> 7) & 255) <= 254
            # Exact rational equality certifies representability; no rounding
            # oracle is needed because these products have exactly these words.
            product = value(a) * value(b)
            assert product == value(out)
            products.append(str(product))
            input_xor ^= a | (b << 16)
            output_xor ^= out
        assert input_xor == 0
        assert output_xor == 1 << j
        records.append({
            "bit": j,
            "inputs_hex": [[f"{a:04x}", f"{b:04x}"] for a, b in inputs],
            "output_hex": [f"{out:04x}" for out in outputs],
            "exact_product_values": products,
            "input_xor_hex": f"{input_xor:08x}",
            "second_difference_hex": f"{output_xor:04x}",
        })
    assert rank([int(r["second_difference_hex"], 16) for r in records]) == 15
    sources = [
        "experiments/native_global_transition_20260908/results/probe_graph/graph_0_before.txt",
        "experiments/native_global_transition_20260908/results/probe_graph/graph_0.json",
        "experiments/paid_krylov_feedback_20261001/NATIVE_EXTENSION_GATE.md",
        "experiments/paid_krylov_feedback_20261001/REPORT.md",
    ]
    result = {
        "kind": "exact rational/bit certificate, not native runtime measurement",
        "word_format": "BF16",
        "scalar_quadruples": 15,
        "scalar_second_difference_rank": 15,
        "width_extension": "DERIVED: direct sum gives 15*m for every positive integer m",
        "witness_rounding": "none: each nonnegative normal/zero product is exact",
        "sign_upper_bound": "symbolic IEEE sign-XOR proof in REPORT.md; not an exhaustive runtime test",
        "native_model_or_backend_run": False,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "witnesses": records,
    }
    output = HERE / "certificate.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: 15 exact legal scalar quadruples; GF(2) rank 15; no floating-point execution")


if __name__ == "__main__":
    main()
