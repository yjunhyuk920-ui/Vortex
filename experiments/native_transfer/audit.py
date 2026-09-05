"""Deterministic local validation; all generated inputs are synthetic."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import math
import random
import struct
import sys
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_transfer import (PhaseSummary, PrefixIndex, bits, value, fma, exact_fma_word,
    reference_dot, summarize, suffix_dot, rank_word, unrank_word, preimage,
    MIN_FINITE_RANK, MAX_FINITE_RANK)

ROOT = Path(__file__).resolve().parents[2]


def bf16(rng: random.Random) -> float:
    return (-1 if rng.randrange(2) else 1) * math.ldexp(rng.randrange(128, 256) / 128.0, rng.randrange(-4, 5))


def word_hash(vectors: list[list[float]]) -> str:
    h = hashlib.sha256()
    for row in vectors:
        h.update(struct.pack('<Q', len(row)))
        for v in row:
            h.update(struct.pack('<I', bits(v)))
    return h.hexdigest()


def run() -> tuple[dict, list[dict]]:
    prereg = ROOT / 'experiments/native_transfer/preregistration.json'
    config = json.loads(prereg.read_text())
    rng = random.Random(config['datasets']['seed'])
    native_mismatches = 0
    native_cases = config['datasets']['native_oracle_cases']
    for _ in range(native_cases):
        ops = []
        for _ in range(3):
            word = rng.getrandbits(32)
            if word & 0x7F800000 == 0x7F800000:
                word ^= 1 << 23
            ops.append(value(word))
        native_mismatches += bits(fma(*ops)) != exact_fma_word(*ops)
    # Signed zeros, subnormals, cancellations, maximum finite, overflow, FMA separation.
    boundary_ops = [
        (value(0x80000000), 1.0, value(0x80000000)),
        (value(1), 0.5, 0.0), (value(3), 0.5, 0.0),
        (value(0x7F7FFFFF), 2.0, -value(0x7F7FFFFF)),
        (value(0x7F7FFFFF), 2.0, value(0x7F7FFFFF)),
        (value(0x3F800001), value(0x3F800001), -value(0x3F800002)),
        (1.0, -1.0, 1.0), (-1.0, 1.0, 1.0)
    ]
    for ops in boundary_ops:
        native_mismatches += bits(fma(*ops)) != exact_fma_word(*ops)
    phase_accepted, phase_mismatch, associativity_mismatch = 0, 0, 0
    for _ in range(config['datasets']['phase_cases']):
        e = rng.randrange(-110, 100)
        sign = -1 if rng.randrange(2) else 1
        k = rng.randrange((1 << 23) + 50000, (1 << 24) - 50000)
        initial = value(bits(math.ldexp(sign * k, e - 23)))
        n = rng.randrange(1, 65)
        w = [value(bits(math.ldexp(rng.randrange(-31, 32) / (1 << rng.randrange(0, 8)), e - 23))) for _ in range(n)]
        x = [1.0] * n
        s = summarize(w, x, e)
        sequential = summarize(w, x, e, balanced=False)
        associativity_mismatch += s != sequential
        split1, split2 = n // 3, 2 * n // 3
        a, b, c = (summarize(w[l:r], x[l:r], e) for l, r in ((0,split1),(split1,split2),(split2,n)))
        associativity_mismatch += a.then(b).then(c) != a.then(b.then(c))
        ans = s.apply(initial, e)
        if ans is not None:
            phase_accepted += 1
            phase_mismatch += bits(ans) != bits(reference_dot(w, x, initial))
    cw, cx, initial = [1.0, 2.0**-23, -1.0], [1.0]*3, 1.5
    s = summarize(cw, cx, 0)
    k = int(initial / 2.0**-23)
    naive = math.ldexp(k + s.delta[k & 1], -23)
    crossing = {'native_word':bits(reference_dot(cw, cx, initial)), 'unguarded_word':bits(naive), 'guard_rejected':s.apply(initial,0) is None}
    inverse_mismatches, max_inverse_calls, inverse_checked = 0, 0, 0
    for _ in range(config['datasets']['inverse_cases']):
        w, x, a = bf16(rng), bf16(rng), bf16(rng)
        target = bits(fma(w,x,a))
        r = rank_word(target)
        inv = preimage(w,x,r,r)
        if inv is None:
            inverse_mismatches += 1
            continue
        lo, hi, calls = inv
        max_inverse_calls = max(max_inverse_calls,calls)
        for z in (lo, hi, (lo+hi)//2, rank_word(bits(a))):
            inverse_checked += 1
            inverse_mismatches += not lo <= z <= hi or bits(fma(w,x,value(unrank_word(z)))) != target
        for z in (lo-1,hi+1):
            if MIN_FINITE_RANK <= z <= MAX_FINITE_RANK:
                inverse_checked += 1
                inverse_mismatches += bits(fma(w,x,value(unrank_word(z)))) == target
    rows = []
    def case(family: str, w: list[float], x: list[float], ordinal: int) -> None:
        index = PrefixIndex.build(w, config['datasets']['suffix_initial_width'])
        out,c = suffix_dot(index,x)
        native = reference_dot(w,x)
        row = {'family':family,'ordinal':ordinal,'n':len(w),'input_sha256':word_hash([w,x]),
               'native_word':bits(native),'output_word':bits(out),'bitwise_equal':bits(native)==bits(out),
               'metadata_bytes':index.metadata_bytes,'constructor_weight_reads':index.constructor_weight_reads,
               **dataclasses.asdict(c),'total_fmas':c.total_fmas,
               'logical_distinct_weight_fraction':c.distinct_weight_reads/len(w),
               'fma_work_fraction':c.total_fmas/len(w),
               'scope':'synthetic IEEE binary32 serial FMA; not a public-model forward or physical traffic measurement'}
        rows.append(row)
    for n in config['datasets']['widths']:
        for i in range(config['datasets']['random_cases_per_width']):
            case('independent_bf16', [bf16(rng) for _ in range(n)], [bf16(rng) for _ in range(n)], i)
    case('suffix_erasure_positive', [1.0]*1022+[2.0**80,-2.0**80], [1.0]*1024, 0)
    case('alternating_no_erasure', [1.0,-1.0]*512, [1.0]*1024, 0)
    normal = [r for r in rows if r['family']=='independent_bf16']
    summary = {
        'version':'NTC-2026-09-05-v1','base_commit':config['base_commit'],
        'preregistration_sha256':hashlib.sha256(prereg.read_bytes()).hexdigest(),
        'native_integer_oracle':{'cases':native_cases+len(boundary_ops),'bit_mismatches':native_mismatches},
        'phase_summary':{'cases':config['datasets']['phase_cases'],'guard_accepted':phase_accepted,'native_bit_mismatches':phase_mismatch,'composition_mismatches':associativity_mismatch},
        'binade_crossing_counterexample':crossing,
        'inverse_preimage':{'cases':config['datasets']['inverse_cases'],'membership_and_boundary_checks':inverse_checked,'mismatches':inverse_mismatches,'maximum_native_calls_per_stage':max_inverse_calls},
        'suffix_evaluator':{'synthetic_cases':len(rows),'bit_mismatches':sum(not r['bitwise_equal'] for r in rows),
                            'ordinary_cases':len(normal),'ordinary_certified':sum(r['certified'] for r in normal),
                            'ordinary_distinct_weight_fraction_min':min(r['logical_distinct_weight_fraction'] for r in normal),
                            'ordinary_fma_work_fraction_min':min(r['fma_work_fraction'] for r in normal),
                            'ordinary_fma_work_fraction_max':max(r['fma_work_fraction'] for r in normal)},
        'positive_control':rows[-2], 'negative_control':rows[-1],
        'paid_budget_conclusion': 'Phase builder still reads every product; inverse builder is query dependent; suffix construction has no target-wide bound on late or non-coalescing rows. No >=10x fully charged core or O5 closure.',
        'decision':'SCOPED_NATIVE_TRANSFER_CONSTRUCTION_VERIFIED_NO_CORE_PROMOTED',
        'THEORY_STATUS':'NOT_ESTABLISHED','HARDWARE_STATUS':'NOT_TESTED',
        'O1_O6_full_mission':'OPEN; scoped native lemmas are not full mission closures',
        'public_checkpoint_execution':'NOT PERFORMED','full_repository_suite':'NOT RUN',
        'physical_memory_latency':'NOT MEASURED','actions_dispatched':False,
        'novelty':'No priority claim; exact IEEE semantics and monotone interval principles are established background.'
    }
    return summary, rows

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'results/native_transfer')
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    summary,rows=run()
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (args.out/'rows.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))
    print(json.dumps(summary,indent=2,sort_keys=True))
