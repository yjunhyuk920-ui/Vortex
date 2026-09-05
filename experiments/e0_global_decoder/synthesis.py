"""Portable bounded QF_BV synthesis. UNKNOWN is never converted to rejection.

Uses a local libz3, with no package installation, network, or hosted job.
The truth-table variables describe an arbitrary NON-systematic encoding. This
is a tiny prerequisite only: it neither compiles a checkpoint nor charges a
native decoder. Default operation emits the formula; --solve runs it.
"""
from __future__ import annotations
import argparse
import ctypes as C
import ctypes.util
import hashlib
import json
import time
from pathlib import Path


def build_smt(rows=2, cols=3, cells=7, timeout_ms=30000, fixed_cells=None):
    if rows * cols > 6 or cells < 3 or cells > 7 or timeout_ms <= 0:
        raise ValueError("only the registered tiny finite domain is supported")
    width = 1 << (rows * cols)
    masks = [sum((((left >> i) & 1) and ((right >> j) & 1)) << (i * cols + j)
                 for i in range(rows) for j in range(cols))
             for left in range(1, 1 << rows) for right in range(1, 1 << cols)]
    p = ['(set-option :produce-models true)', f'(set-option :timeout {timeout_ms})',
         '(set-option :smt.random_seed 0)', '(set-logic QF_BV)']
    for j in range(cells):
        p += [f'(declare-const C{j} (_ BitVec {width}))',
              f'(assert (= ((_ extract 0 0) C{j}) #b0))',
              f'(assert (not (= C{j} (_ bv0 {width}))))']
    p += ['(assert (distinct ' + ' '.join('C'+str(i) for i in range(cells)) + '))']
    for j in range(3, cells-1):
        p += [f'(assert (bvult C{j} C{j+1}))']
    expr = f'C{cells-1}'
    for j in range(cells-2, -1, -1):
        expr = f'(ite (= i (_ bv{j} 3)) C{j} {expr})'
    p += [f'(define-fun pick ((i (_ BitVec 3))) (_ BitVec {width}) {expr})',
          f'(define-fun mask ((b Bool)) (_ BitVec {width}) (ite b (_ bv{(1 << width)-1} {width}) (_ bv0 {width})))']
    for q, m in enumerate(masks):
        for name in ['a','b','c']:
            p += [f'(declare-const {name}{q} (_ BitVec 3))',
                  f'(assert (bvule {name}{q} (_ bv{cells-1} 3)))']
        for name in ['u','v','w']:
            p += [f'(declare-const {name}{q} Bool)']
        p += [f'(assert (distinct a{q} b{q}))', f'(assert (distinct a{q} c{q}))']
        expression = f'(bvor (bvand (bvnot (pick a{q})) (pick b{q}) (mask u{q})) (bvand (pick a{q}) (bvor (bvand (bvnot (pick c{q})) (mask v{q})) (bvand (pick c{q}) (mask w{q})))))'
        truth = sum(((x & m).bit_count() % 2) << x for x in range(width))
        p += [f'(assert (= {expression} (_ bv{truth} {width})))']
    p += ['(assert (= a0 (_ bv0 3)))', '(assert (= b0 (_ bv1 3)))',
          '(assert (or (= c0 (_ bv1 3)) (= c0 (_ bv2 3))))']
    if fixed_cells is not None:
        if len(fixed_cells) != cells:
            raise ValueError("wrong fixed encoder length")
        p += [f'(assert (= C{i} (_ bv{value} {width})))' for i, value in enumerate(fixed_cells)]
    p += ['(check-sat)', '(get-info :reason-unknown)']
    return '\n'.join(p) + '\n'


def solve(text: str) -> dict:
    name = ctypes.util.find_library('z3')
    if not name:
        raise RuntimeError('libz3 unavailable; missing solver is not a scientific rejection')
    z = C.CDLL(name)
    z.Z3_mk_config.restype = C.c_void_p
    z.Z3_mk_context.argtypes = [C.c_void_p]
    z.Z3_mk_context.restype = C.c_void_p
    z.Z3_del_config.argtypes = [C.c_void_p]
    z.Z3_del_context.argtypes = [C.c_void_p]
    z.Z3_eval_smtlib2_string.argtypes = [C.c_void_p, C.c_char_p]
    z.Z3_eval_smtlib2_string.restype = C.c_char_p
    z.Z3_get_full_version.restype = C.c_char_p
    cfg = z.Z3_mk_config()
    ctx = z.Z3_mk_context(cfg)
    z.Z3_del_config(cfg)
    start = time.perf_counter()
    try:
        output = z.Z3_eval_smtlib2_string(ctx, text.encode()).decode()
    finally:
        z.Z3_del_context(ctx)
    first = output.splitlines()[0].strip()
    if first not in ('sat', 'unsat', 'unknown'):
        raise RuntimeError('solver did not produce a recognized status: ' + output)
    return {'solver_version': z.Z3_get_full_version().decode(),
            'formula_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'result': first, 'raw_stdout': output,
            'elapsed_seconds': time.perf_counter()-start,
            'classification': 'INCONCLUSIVE' if first == 'unknown' else 'REQUIRES_INDEPENDENT_WITNESS_OR_PROOF_REVIEW'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--solve', action='store_true')
    p.add_argument('--write-input', type=Path)
    args = p.parse_args()
    text = build_smt()
    if args.write_input:
        args.write_input.write_text(text, encoding='utf-8')
    if args.solve:
        print(json.dumps(solve(text), indent=2, sort_keys=True))
    else:
        print(hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main()
