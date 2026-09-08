"""Rebuild real-model programs and compare frozen numerical/code evidence.

Timings are retained in original files but excluded from deterministic comparison.
No reference hashes are overwritten by this verifier.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parent
VOLATILE={'build_seconds','load_seconds_observed','demand_constructor_seconds'}

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in VOLATILE}
    if isinstance(x,list):return [clean(v) for v in x]
    return x

def manifest(folder):
    out={}
    for p in sorted(Path(folder).rglob('*')):
        if not p.is_file() or p.name=='audit.json':continue
        data=p.read_bytes()
        if p.suffix=='.json':data=json.dumps(clean(json.loads(data)),sort_keys=True,separators=(',',':')).encode()
        out[str(p.relative_to(folder)).replace('\\','/')]=hashlib.sha256(data).hexdigest()
    return out

def main():
    a=argparse.ArgumentParser();a.add_argument('--freeze',action='store_true');a.add_argument('--replay',type=Path)
    a.add_argument('--receipt',type=Path,help='New receipt path; existing receipts are never overwritten')
    args=a.parse_args();baseline=ROOT/'results/run_v3';frozen=ROOT/'results/frozen_manifest.json'
    if args.freeze:
        if frozen.exists():raise RuntimeError('Refuse replacing frozen baseline')
        frozen.write_text(json.dumps(manifest(baseline),sort_keys=True,indent=2)+'\n',encoding='utf-8')
        print('Frozen',len(manifest(baseline)),'science files');return
    target=(args.receipt.resolve() if args.receipt else ROOT/'results/validation.json')
    if target.exists() and args.receipt is None and args.replay:
        target=ROOT/'results'/('validation_'+args.replay.name+'.json')
    if target.exists():raise RuntimeError('Receipt already exists; specify a fresh --receipt')
    if args.replay and args.replay.resolve().exists():raise RuntimeError('Replay destination must not exist')
    expect=json.loads(frozen.read_text());assert manifest(baseline)==expect
    subprocess.run([sys.executable,'-m','unittest','-v','test_transition'],cwd=ROOT,check=True)
    receipt={'frozen_file_count':len(expect),'existing_science_hashes_match':True,
             'unit_tests_executed':True,'replay_executed':False,
             'volatile_fields':sorted(VOLATILE),'timings_not_claimed_reproducible':True}
    if args.replay:
        destination=args.replay.resolve()
        if destination.exists():raise RuntimeError('Replay destination must not exist')
        subprocess.run([sys.executable,'run_transition.py','--explicit-mask','--out',str(destination)],cwd=ROOT,check=True)
        got=manifest(destination);missing=sorted(set(expect)-set(got));extra=sorted(set(got)-set(expect))
        changed=sorted(k for k in expect.keys()&got.keys() if expect[k]!=got[k])
        receipt.update(replay_executed=True,replay_file_count=len(got),missing=missing,extra=extra,changed=changed)
        if missing or extra or changed:
            target.write_bytes((json.dumps(receipt,indent=2)+'\n').encode('utf-8'))
            raise AssertionError('Replay mismatch; baseline remains intact')
    target.write_bytes((json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
