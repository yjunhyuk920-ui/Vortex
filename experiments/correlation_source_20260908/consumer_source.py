"""Exact consumer-specialized XOR source. Not a native model executor.

All construction work is paid. Runtime reads every stored plane word, plus
mask/output coefficient words. Full original source survives outside the query
object. Fixed finite ANFs are compiler inputs, not learned approximations.
"""
from __future__ import annotations
import hashlib
import json
import random
import struct
from pathlib import Path

HEAD = struct.Struct('<8sIIII')
MAGIC = b'VXCS01\0\0'

def value(p: int, terms: list[int]) -> int:
    r = 0
    for t in terms:
        r ^= int(p & t == t)
    return r

def compile_consumers(words: list[int], k: int, functions: list[list[int]]):
    if type(k) is not int or not 1 <= k <= 4096 or not words or not functions:
        raise ValueError('empty/invalid domain')
    if any(type(p) is not int or not 0 <= p < 1 << k for p in words):
        raise ValueError('source word')
    if any(type(t) is not int or not 0 <= t < 1 << k for f in functions for t in f):
        raise ValueError('ANF')
    if len(words) >= 2**32 or len(functions) >= 2**32:
        raise ValueError('file dimensions')
    patterns = sorted(set(words))
    idx = {p: i for i, p in enumerate(patterns)}
    truth = []
    constants = []
    for f in functions:
        c = value(patterns[0], f)
        constants.append(c)
        truth.append(sum((value(p, f) ^ c) << j for j, p in enumerate(patterns)))
    pivots = {}
    selected = []
    masks = []
    elimination_xors = 0
    for f, original in enumerate(truth):
        v, combination = original, 0
        for pivot in sorted(pivots, reverse=True):
            if v >> pivot & 1:
                row, relation = pivots[pivot]
                v ^= row
                combination ^= relation
                elimination_xors += 2
        if v:
            slot = len(selected)
            selected.append(f)
            pivots[v.bit_length()-1] = (v, combination ^ (1 << slot))
            masks.append(1 << slot)
        else:
            masks.append(combination)
    # Actual certificate check on all observed patterns, not test queries.
    for f, mask in enumerate(masks):
        reconstructed = 0
        for j, basis in enumerate(selected):
            if mask >> j & 1:
                reconstructed ^= truth[basis]
        if reconstructed != truth[f]:
            raise AssertionError('invalid compiler relation')
    n, nf, rank = len(words), len(functions), len(selected)
    nb, rb = (n+7)//8, (rank+7)//8
    body = bytearray(HEAD.pack(MAGIC, n, nf, rank, k))
    body.extend(bytes(constants))
    for mask in masks:
        body.extend(mask.to_bytes(rb, 'little'))
    for f in selected:
        plane = sum(((truth[f] >> idx[p]) & 1) << i for i, p in enumerate(words))
        body.extend(plane.to_bytes(nb, 'little'))
    data = bytes(body) + hashlib.sha256(body).digest()
    certificate = dict(patterns=patterns, centered_rows=truth, constants=constants,
                       selected_function_indices=selected, coefficient_masks=masks,
                       rank=rank, constructor_source_words=n,
                       anf_evaluations=(len(patterns)+1)*nf,
                       elimination_bigint_xors=elimination_xors,
                       compiled_plane_bits=n*rank,
                       original_bits=n*k, file_bytes=len(data))
    return data, certificate

class ConsumerSource:
    def __init__(self, data: bytes):
        if len(data) < HEAD.size+32:
            raise ValueError('short')
        body = data[:-32]
        if hashlib.sha256(body).digest() != data[-32:]:
            raise ValueError('checksum')
        magic, self.n, self.f, self.r, self.k = HEAD.unpack_from(body)
        if magic != MAGIC or not self.n or not self.f or self.r > self.f or not 1 <= self.k <= 4096:
            raise ValueError('header')
        self.nb, self.rb = (self.n+7)//8, (self.r+7)//8
        self.mask_start = HEAD.size + self.f
        self.plane_start = self.mask_start + self.f*self.rb
        if len(body) != self.plane_start + self.r*self.nb:
            raise ValueError('length')
        if any(c > 1 for c in body[HEAD.size:self.mask_start]):
            raise ValueError('constants')
        self.data = data
        for i in range(self.f):
            if self._mask(i) >= 1 << self.r:
                raise ValueError('coefficient')
        for i in range(self.r):
            start = self.plane_start+i*self.nb
            if int.from_bytes(body[start:start+self.nb], 'little') >= 1 << self.n:
                raise ValueError('padding')
        # This full loading parse/check is not a free query.

    def _mask(self, i):
        pos = self.mask_start+i*self.rb
        return int.from_bytes(self.data[pos:pos+self.rb], 'little')

    def query(self, mask: int):
        if type(mask) is not int or not 0 <= mask < 1 << self.n:
            raise ValueError('query')
        query_bytes = mask.to_bytes(self.nb, 'little')
        count_parity = sum(b.bit_count() for b in query_bytes) & 1
        answers = 0
        chunks = (self.nb+7)//8
        for j in range(self.r):
            acc = 0
            start = self.plane_start+j*self.nb
            for offset in range(0, self.nb, 8):
                take = min(8, self.nb-offset)
                a = int.from_bytes(self.data[start+offset:start+offset+take], 'little')
                q = int.from_bytes(query_bytes[offset:offset+take], 'little')
                acc ^= (a & q).bit_count() & 1
            answers |= acc << j
        output = [self.data[HEAD.size+i]*count_parity ^ (self._mask(i) & answers).bit_count() % 2
                  for i in range(self.f)]
        cost = dict(source_plane_bytes=self.r*self.nb,
                    coefficient_bytes=self.f*self.rb, constant_bytes=self.f,
                    query_mask_bytes=(self.r+1)*self.nb,
                    output_bytes=self.f, plane_and_popcounts=self.r*chunks,
                    output_popcount_words=self.f*((self.r+63)//64),
                    total_named_byte_references=self.r*self.nb + self.f*self.rb +
                        self.f + (self.r+1)*self.nb + self.f,
                    scope='Named fields only; Python/addresses/code/allocations and hardware service costs remain')
        return output, cost

def check_direct(words, functions, q):
    # Written separately as individual monomial products and integer sums.
    out = []
    for terms in functions:
        acc = 0
        for i, p in enumerate(words):
            if q >> i & 1:
                for term in terms:
                    a = term
                    product = 1
                    while a:
                        bit = a & -a
                        product *= int(bool(p & bit))
                        a ^= bit
                    acc += product
        out.append(acc % 2)
    return out

def run(out: Path, old: Path):
    if out.exists():
        raise ValueError('no overwriting results')
    out.mkdir(parents=True)
    cases = json.loads((old/'inputs.json').read_text(encoding='utf-8'))
    fixtures = []
    for case in cases:
        masks = []
        for rm in range(16):
            for cm in range(16):
                masks.append(sum(((rm>>i&1) * (cm>>j&1)) << (4*i+j)
                                 for i in range(4) for j in range(4)))
        fixtures.append((case['name'], case['values'], case['k'], case['functions'], masks, 'prior_corpus'))
    for k in [8,16,32]:
        for seed in [11,29]:
            rng = random.Random(seed)
            words = [0]+[1<<j for j in range(k)]
            words += [rng.randrange(1<<k) for _ in range(1024-len(words))]
            f = [(1<<j)|(1<<(j+1)) for j in range(0,k,2)]
            masks = [0,(1<<1024)-1,1,sum(1<<j for j in range(0,1024,2))]
            masks += [rng.getrandbits(1024) for _ in range(12)]
            for name, fs in [('single',[f]),('retained',[[1<<j] for j in range(k)]+[f])]:
                fixtures.append((f'k{k}_s{seed}_{name}',words,k,fs,masks,'continuation_fixed'))
    summaries = []
    with (out/'raw.jsonl').open('w',encoding='utf-8',newline='\n') as raw:
        for name, words, k, fs, masks, origin in fixtures:
            data, cert = compile_consumers(words,k,fs)
            (out/(name+'.bin')).write_bytes(data)
            (out/(name+'.json')).write_text(json.dumps(dict(words=words,k=k,functions=fs,
                      masks=masks,certificate=cert,origin=origin),sort_keys=True)+'\n',encoding='utf-8')
            src = ConsumerSource(data)
            for qi, q in enumerate(masks):
                actual,cost = src.query(q)
                expected = check_direct(words,fs,q)
                if actual != expected:
                    raise AssertionError(name)
                raw.write(json.dumps(dict(case=name,query=qi,actual=actual,expected=expected,
                                          cost=cost),sort_keys=True)+'\n')
            summaries.append(dict(case=name,origin=origin,n=len(words),k=k,f=len(fs),rank=src.r,
                   file_bytes=len(data),original_packed_bytes=(len(words)*k+7)//8,
                   queries=len(masks),output_bits=len(masks)*len(fs),full_query_cost=src.query((1<<len(words))-1)[1]))
    summary = dict(cases=summaries,queries=sum(c['queries'] for c in summaries),
                   output_bits=sum(c['output_bits'] for c in summaries),mismatches=0,
                   newly_closed_full_mission_obligations=0,theory_status='NOT_ESTABLISHED',
                   hardware_status='NOT_TESTED',native_order_execution='NOT TESTED',full_model='NOT TESTED')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return summary

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--old-results',type=Path,required=True)
    args = p.parse_args()
    result = run(args.output,args.old_results)
    print(json.dumps({key:val for key,val in result.items() if key != 'cases'},sort_keys=True))
