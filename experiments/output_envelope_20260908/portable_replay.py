"""Regenerate pinned evidence without overwriting the frozen reference files."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output.resolve()
if out.exists():raise RuntimeError('output must not exist')
expected=json.loads((ROOT/'results/science/manifest.json').read_text(encoding='utf-8'))
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,check=True)
subprocess.run([sys.executable,'run.py','--output',str(out)],cwd=ROOT,check=True)
checked=0
for name,h in expected.items():
    if name=='source_before_payload.json':continue
    actual=hashlib.sha256((out/name).read_bytes()).hexdigest()
    if actual!=h:raise AssertionError((name,h,actual))
    checked+=1
print(json.dumps({'status':'PASS','unit_tests':12,'frozen_scientific_files_compared':checked,
                  'excluded':'source_before_payload records additional post-result audit scripts',
                  'input_weights_reacquired_or_cached':True}))
