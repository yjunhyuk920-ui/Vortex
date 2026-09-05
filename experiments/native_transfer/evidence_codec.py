"""Lossless predictor/residual codec for the two large recorded inverse files.

Decoding is NOT a new native experiment. The predictor executes the independent
integer IEEE specification, never fmaf. Literal residuals preserve any original
observation that differs. Original byte hashes are mandatory on decode.
The unencoded native captures are also included in the downloadable local bundle.
"""
from __future__ import annotations
import argparse,hashlib,importlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import native_transfer as nt
ROOT=Path(__file__).resolve().parents[2]
CODEC=ROOT/'results/native_transfer/inverse_recording.json'

def encode_json(data):
    return (json.dumps(data,indent=2,sort_keys=True)+'\n').encode()

def predictions():
    old=nt.fma
    nt.fma=lambda w,x,a:nt.value(nt.exact_fma_word(w,x,a))
    try:
        audit=importlib.import_module('audit_inverse')
        # audit_inverse copies fma with import *, so update its local name too.
        old_audit=audit.fma
        audit.fma=nt.fma
        try:
            _,rows,suffix=audit.run()
        finally:
            audit.fma=old_audit
        return {'inverse_rows.json':rows,'inverse_suffix_rows.json':suffix}
    finally:
        nt.fma=old

def encode():
    pred=json.loads(encode_json(predictions())); result={'codec':'integer-IEEE-predictor-plus-literal-record-residual-v1',
        'meaning':'Lossless encoding of the previously captured native records. Decode is not a new native run.',
        'dependencies':{},'files':{}}
    for name in ('native_transfer.py','audit_inverse.py','continuation.json','evidence_codec.py'):
        p=ROOT/'experiments/native_transfer'/name
        result['dependencies'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for name,records in pred.items():
        original=(ROOT/'results/native_transfer'/name).read_bytes()
        observed=json.loads(original)
        if len(observed)!=len(records):raise ValueError('record count mismatch')
        residuals={str(i):r for i,r in enumerate(observed) if r!=records[i]}
        result['files'][name]={'records':len(observed),'sha256':hashlib.sha256(original).hexdigest(),
                              'literal_residuals':residuals}
    CODEC.write_bytes(encode_json(result))
    return result

def decode(out):
    record=json.loads(CODEC.read_bytes())
    for name,expected in record['dependencies'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('codec dependency changed: '+name)
    pred=predictions(); out.mkdir(parents=True,exist_ok=True)
    receipts={}
    for name,spec in record['files'].items():
        rows=pred[name]
        if len(rows)!=spec['records']:raise ValueError('count mismatch')
        for i,r in spec['literal_residuals'].items():rows[int(i)]=r
        b=encode_json(rows); h=hashlib.sha256(b).hexdigest()
        if h!=spec['sha256']:raise ValueError('record hash mismatch: '+name)
        (out/name).write_bytes(b)
        receipts[name]={'byte_identical':True,'sha256':h,'bytes':len(b)}
    return receipts

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--encode',action='store_true')
    parser.add_argument('--out',type=Path,default=ROOT/'results/native_transfer/decoded')
    args=parser.parse_args()
    print(json.dumps(encode() if args.encode else decode(args.out),indent=2,sort_keys=True))
