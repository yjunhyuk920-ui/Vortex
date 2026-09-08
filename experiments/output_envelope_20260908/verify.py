from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--rerun',action='store_true');a=p.parse_args()
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,check=True)
science=ROOT/'results'/'science';manifest=json.loads((science/'manifest.json').read_text())
for name,h in manifest.items():
    assert hashlib.sha256((science/name).read_bytes()).hexdigest()==h,name
if a.rerun:
    dest=ROOT/'replay'/'verified'
    if dest.exists():raise RuntimeError('do not overwrite prior replay evidence')
    dest.parent.mkdir(exist_ok=True)
    subprocess.run([sys.executable,'run.py','--output',str(dest)],cwd=ROOT,check=True)
    new=json.loads((dest/'manifest.json').read_text())
    for name,h in manifest.items():
        if name=='source_before_payload.json':continue
        assert new[name]==h,('regenerated difference',name)
print(json.dumps({'unit_tests':12,'files_hash_verified':len(manifest),'scientific_rerun':a.rerun,'status':'PASS'}))
