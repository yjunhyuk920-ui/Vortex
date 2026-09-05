"""Deterministic bounded source audit. No timings or LLM/target claims."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from experiments.query_source.partition_source import build, query, reference, bits

ROOT = Path(__file__).resolve().parents[2]


class RNG:
    def __init__(self, seed): self.s = seed & 0xFFFFFFFF
    def next(self):
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.s = x & 0xFFFFFFFF
        return self.s


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def case(kind, n):
    r = RNG(20260905 + n)
    W = [[bits(((r.next() % 2049) - 1024) / 128) for j in range(n)] for i in range(n)]
    X = [[bits(((r.next() % 2049) - 1024) / 256) for j in range(n)] for k in range(4)]
    if kind == "engineered_absorption_positive_control":
        for row in W: row[0] = bits(1.0)
        X = [[bits(2.0 ** 30)] + [bits(((r.next() % 2049) - 1024) / 1024)
                                        for j in range(n - 1)] for k in range(4)]
    elif kind == "unique_column_adversary":
        W = [[bits(1 + (i + j * n) / (2 * n * n)) for j in range(n)] for i in range(n)]
        X = [[bits(1.0)] * n for _ in range(4)]
    elif kind != "ordinary_dyadic":
        raise ValueError(kind)
    return W, X


def main():
    spec_bytes = (ROOT / "experiments/query_source/preregistration.json").read_bytes()
    spec = json.loads(spec_bytes)
    records = []
    for kind in spec["populations"]:
        for n in spec["sizes"]:
            W, X = case(kind, n)
            idx = build(W)
            encoded = idx.encode()
            assert len(encoded) == 32 + 12 * n + sum(len(c.buckets) * (4 + 8 * idx.mask_words) for c in idx.columns)
            results = []
            for x in X:
                actual, cnt = query(idx, x)
                expected = reference(W, x)
                assert len(actual) == len(expected) == n
                mismatch = sum(a != b for a, b in zip(actual, expected))
                assert mismatch == 0
                results.append({"input_sha256": digest(x), "output_words": actual,
                                "reference_sha256": digest(expected), "mismatches": mismatch,
                                "counts": cnt.json()})
            records.append({"population": kind, "rows": n, "cols": n,
                            "matrix_sha256": digest(W), "constructor_weight_reads": n * n,
                            "original_fp32_bytes": 4 * n * n,
                            "serialized_index_bytes": len(encoded),
                            "serialized_index_sha256": hashlib.sha256(encoded).hexdigest(),
                            "native_fmas_per_query": n * n, "queries": results})
    out = {"preregistration_sha256": hashlib.sha256(spec_bytes).hexdigest(),
           "decision": "NO_UNIVERSAL_CHEAP_SOURCE_CONSTRUCTED",
           "theory_status": "NOT_ESTABLISHED", "hardware_status": "NOT_TESTED",
           "core_admission": False, "actual_transformer_replacement": "NOT_TESTED",
           "logical_metrics_are_not_hardware": True, "records": records}
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__": main()
