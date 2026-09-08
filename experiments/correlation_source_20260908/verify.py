"""Independent file decoder and direct Boolean checks; no source.py import."""
import hashlib
import json
import struct
from pathlib import Path


def verify(root: Path) -> dict:
    cases = {v['name']: v for v in json.loads((root/'inputs.json').read_text(encoding='utf-8'))}
    for name, case in cases.items():
        data = (root/(name+'.bin')).read_bytes()
        body = data[:-32]
        assert hashlib.sha256(body).digest() == data[-32:]
        magic, m, n, k, r, w, flags = struct.unpack_from('<8sIIIIII', body)
        assert magic == b'VXCRR01\0' and flags == 0
        assert (m, n, k) == (case['rows'], case['cols'], case['k'])
        step = (k+7)//8
        pats = [int.from_bytes(body[32+a*step:32+(a+1)*step], 'little') for a in range(r)]
        # Decode complete bitstream independently of Source.identifier slices.
        stream = int.from_bytes(body[32+r*step:], 'little')
        values = [pats[(stream >> (j*w)) & ((1 << w)-1)] for j in range(m*n)]
        assert values == case['values']
    count = bits = 0
    seen = set()
    with (root/'raw.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            case = cases[row['case']]
            ident = (row['case'], row['rowmask'], row['colmask'])
            assert ident not in seen
            seen.add(ident)
            positions = [i*case['cols']+j for i in range(case['rows'])
                         for j in range(case['cols'])
                         if row['rowmask'] & (1 << i) and row['colmask'] & (1 << j)]
            outputs = []
            for monomials in case['functions']:
                total = 0
                for pos in positions:
                    p = case['values'][pos]
                    for monomial in monomials:
                        # Separate product-of-bits reference for each ANF term.
                        total += all(p & (1 << b) for b in range(case['k']) if monomial & (1 << b))
                outputs.append(total % 2)
            assert outputs == row['actual'] == row['expected']
            assert row['cost']['selected_sites'] == len(positions)
            count += 1
            bits += len(outputs)
    assert count == len(cases)*256
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    assert summary['totals'] == dict(queries=count, output_bits=bits, mismatches=0)
    return dict(cases=len(cases), queries=count, output_bits=bits, mismatches=0,
                independent_parser=True, independent_boolean_evaluator=True,
                native_reference_scope='Integer four-term witness only; NOT full HF/CUDA')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(verify(a.results), indent=2, sort_keys=True)+'\n', encoding='utf-8')
