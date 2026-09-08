"""Paid joint-correlation constructor. Research primitive, NOT a model executor.

Only stdlib; finite bit tuples are immutable checkpoint data. No source arrays
are retained by the query object. IDs, dictionary reads and ANF work are charged.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import struct
from bisect import bisect_left
from dataclasses import dataclass, asdict

HEADER = struct.Struct('<8sIIIIII')
MAGIC = b'VXCRR01\0'


def packed_size(n: int, width: int) -> int:
    return (n * width + 7) // 8


def pack_ids(ids: list[int], width: int) -> bytes:
    out = bytearray(packed_size(len(ids), width))
    for pos, val in enumerate(ids):
        if not 0 <= val < 1 << width:
            raise ValueError('ID does not fit')
        for j in range(width):
            out[(pos * width + j) // 8] |= ((val >> j) & 1) << ((pos * width + j) % 8)
    return bytes(out)


def compile_source(values: list[int], rows: int, cols: int, k: int) -> bytes:
    if any(type(a) is not int for a in [rows, cols, k]):
        raise ValueError('integer dimensions required')
    if not 1 <= rows < 2**32 or not 1 <= cols < 2**32 or not 1 <= k <= 4096 or len(values) != rows * cols:
        raise ValueError('shape/alphabet')
    if any(type(v) is not int or not 0 <= v < 1 << k for v in values):
        raise ValueError('not a k-bit word')
    ordered = sorted(values)
    patterns = [v for i, v in enumerate(ordered) if i == 0 or v != ordered[i - 1]]
    if len(patterns) >= 2**32:
        raise ValueError('dictionary exceeds file format')
    width = (len(patterns) - 1).bit_length()
    body = HEADER.pack(MAGIC, rows, cols, k, len(patterns), width, 0)
    body += b''.join(v.to_bytes((k + 7) // 8, 'little') for v in patterns)
    body += pack_ids([bisect_left(patterns, v) for v in values], width)
    return body + hashlib.sha256(body).digest()


@dataclass
class Count:
    selected_sites: int = 0
    id_bytes: int = 0
    id_bit_tests: int = 0
    mask_bit_tests: int = 0
    histogram_updates: int = 0
    pattern_bytes: int = 0
    anf_monomials: int = 0
    output_xors: int = 0
    histogram_scan_entries: int = 0


class Source:
    def __init__(self, data: bytes):
        if len(data) < HEADER.size + 32:
            raise ValueError('short source')
        body = data[:-32]
        if hashlib.sha256(body).digest() != data[-32:]:
            raise ValueError('checksum')
        magic, self.rows, self.cols, self.k, self.r, self.width, flags = HEADER.unpack_from(body)
        if magic != MAGIC or flags or not self.rows or not self.cols or not 1 <= self.k <= 4096:
            raise ValueError('header')
        if not 1 <= self.r <= self.rows * self.cols or self.width != (self.r - 1).bit_length():
            raise ValueError('dictionary')
        self.pbytes = (self.k + 7) // 8
        self.id_start = HEADER.size + self.r * self.pbytes
        expected = self.id_start + packed_size(self.rows * self.cols, self.width)
        if len(body) != expected:
            raise ValueError('length')
        self.data = data
        previous = -1
        for i in range(self.r):
            v = self.pattern(i)
            if not previous < v < (1 << self.k):
                raise ValueError('pattern order/range')
            previous = v
        for i in range(self.rows * self.cols):
            if self.identifier(i) >= self.r:
                raise ValueError('invalid ID')
        # The above complete source validation is loading work, not free query work.

    def pattern(self, a: int) -> int:
        start = HEADER.size + a * self.pbytes
        return int.from_bytes(self.data[start:start + self.pbytes], 'little')

    def identifier(self, i: int, count: Count | None = None) -> int:
        if self.width == 0:
            return 0
        bit = i * self.width
        start = self.id_start + bit // 8
        size = (bit % 8 + self.width + 7) // 8
        if count is not None:
            count.id_bytes += size
            count.id_bit_tests += self.width
        word = int.from_bytes(self.data[start:start + size], 'little')
        return (word >> (bit % 8)) & ((1 << self.width) - 1)

    def counts(self, rowmask: int, colmask: int) -> tuple[list[int], Count]:
        if not 0 <= rowmask < 1 << self.rows or not 0 <= colmask < 1 << self.cols:
            raise ValueError('query mask')
        h = [0] * self.r
        cost = Count()
        for i in range(self.rows):
            cost.mask_bit_tests += 1
            if not rowmask >> i & 1:
                continue
            for j in range(self.cols):
                cost.mask_bit_tests += 1
                if colmask >> j & 1:
                    a = self.identifier(i * self.cols + j, cost)
                    h[a] += 1
                    cost.selected_sites += 1
                    cost.histogram_updates += 1
        return h, cost

    def parities(self, rowmask: int, colmask: int,
                 functions: list[list[int]]) -> tuple[list[int], Count]:
        if any(type(m) is not int or not 0 <= m < 1 << self.k for f in functions for m in f):
            raise ValueError('ANF mask')
        hist, cost = self.counts(rowmask, colmask)
        answer = [0] * len(functions)
        for a, multiplicity in enumerate(hist):
            cost.histogram_scan_entries += 1
            if not multiplicity & 1:
                continue
            p = self.pattern(a)
            cost.pattern_bytes += self.pbytes
            for t, terms in enumerate(functions):
                v = 0
                for m in terms:
                    v ^= int((p & m) == m)
                    cost.anf_monomials += 1
                answer[t] ^= v
                cost.output_xors += 1
        return answer, cost


def direct(values: list[int], rows: int, cols: int, rowmask: int,
           colmask: int, functions: list[list[int]]) -> list[int]:
    result = []
    for terms in functions:
        bit = 0
        for i in range(rows):
            for j in range(cols):
                if (rowmask >> i & 1) and (colmask >> j & 1):
                    p = values[i * cols + j]
                    for m in terms:
                        bit ^= int(p & m == m)
        result.append(bit)
    return result


def f32(x: float) -> float:
    return struct.unpack('<f', struct.pack('<f', x))[0]


def native_tree(values: list[float]) -> float:
    """Binary32 witness evaluator; no claim to a complete native HF/CUDA ABI.

    Inputs and intermediates must be finite and within the binary32 range.
    struct.pack rejects overflow rather than implementing CUDA exceptions.
    The independent oracle covers only the exact integer witness below.
    """
    if not values:
        raise ValueError('nonempty reduction required')
    n = 1 << (len(values) - 1).bit_length()
    level = [f32(v) for v in values] + [0.0] * (n - len(values))
    while len(level) > 1:
        level = [f32(level[j] + level[j + 1]) for j in range(0, len(level), 2)]
    return level[0]


def ordered_tree(source: Source) -> tuple[float, dict]:
    """Read already-produced FP32 witness terms; NOT a general native kernel.

    Keeps positional information and original balanced parentheses, but does
    not generate W*x terms cheaply. Numeric scope is native_tree's scope.
    """
    if source.k != 32:
        raise ValueError('FP32 words required')
    values = []
    cost = Count()
    for i in range(source.rows * source.cols):
        a = source.identifier(i, cost)
        b = source.pattern(a)
        cost.pattern_bytes += 4
        values.append(struct.unpack('<f', b.to_bytes(4, 'little'))[0])
    return native_tree(values), dict(asdict(cost), fp32_additions=(1 << (len(values) - 1).bit_length()) - 1)


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    records = []
    inputs = []
    raw = out / 'raw.jsonl'
    with raw.open('w', encoding='utf-8', newline='\n') as stream:
        for k in [2, 4, 8, 16]:
            fs = [[1 << b] for b in range(k)]
            fs += [[(1 << a) | (1 << b)] for a in range(k) for b in range(a + 1, k)]
            fs += [[1 << b for b in range(k)]]
            for seed in [11, 29]:
                rng = random.Random(seed)
                pool = [rng.randrange(1 << k) for _ in range(2)]
                families = {'diverse': [rng.randrange(1 << k) for _ in range(16)],
                            'repeated': [pool[i % 2] for i in range(16)],
                            'constant': [pool[0]] * 16}
                for family, values in families.items():
                    data = compile_source(values, 4, 4, k)
                    name = f'k{k}_s{seed}_{family}'
                    inputs.append(dict(name=name, k=k, rows=4, cols=4, values=values, functions=fs))
                    (out / (name + '.bin')).write_bytes(data)
                    src = Source(data)
                    rec = dict(name=name, k=k, patterns=src.r, id_width=src.width,
                               raw_bits_bytes=packed_size(16, k), file_bytes=len(data),
                               queries=0, output_bits=0, mismatches=0, totals=asdict(Count()))
                    for rm in range(16):
                        for cm in range(16):
                            got, cost = src.parities(rm, cm, fs)
                            ref = direct(values, 4, 4, rm, cm, fs)
                            rec['queries'] += 1
                            rec['output_bits'] += len(got)
                            rec['mismatches'] += sum(a != b for a, b in zip(got, ref))
                            for key, val in asdict(cost).items():
                                rec['totals'][key] += val
                            stream.write(json.dumps(dict(case=name, rowmask=rm, colmask=cm,
                                                         actual=got, expected=ref, cost=asdict(cost)),
                                                    sort_keys=True, separators=(',', ':')) + '\n')
                    _, full = src.parities(15, 15, fs)
                    rec['full_query_cost'] = asdict(full)
                    records.append(rec)
    joint = []
    for values in [[0, 3], [1, 2]]:
        src = Source(compile_source(values, 1, 2, 2))
        got, cost = src.parities(1, 3, [[1], [2], [3]])
        joint.append(dict(source=values, child_parities=got[:2], parent_and=got[2], cost=asdict(cost)))
    native = []
    for values in [[2**24, 1, -2**24, 1], [2**24, -2**24, 1, 1]]:
        bits = [int.from_bytes(struct.pack('<f', v), 'little') for v in values]
        src = Source(compile_source(bits, 1, 4, 32))
        hist, _ = src.counts(1, 15)
        result, cost = ordered_tree(src)
        native.append(dict(terms=values, patterns=[src.pattern(i) for i in range(src.r)],
                           histogram=hist, exact_sum=sum(values), native_result=result,
                           cost=cost, file_bytes=len(src.data)))
    summary = dict(cases=records, joint_witness=joint, native_witness=native,
                   totals={key: sum(r[key] for r in records) for key in ['queries', 'output_bits', 'mismatches']},
                   theory_status='NOT_ESTABLISHED', hardware_status='NOT_TESTED',
                   core_admission=False, public_model='NOT TESTED')
    (out / 'inputs.json').write_text(json.dumps(inputs, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    s = run(args.output)
    print(json.dumps(s['totals'], sort_keys=True))
