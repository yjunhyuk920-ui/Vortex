"""Restore the hash-pinned research sources. No network or implicit execution.
Use --run explicitly to regenerate and check all scientific results in a new folder.
"""
from pathlib import Path
import argparse,base64,lzma,json,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parent
CAPSULE_SHA='8d6effda2ff488ae3c3e84b7ce2eca81e0b56b61f5eb912009ef3a9b9ce5b7f5'
PAYLOAD_SHA='2a58dbe1898c36d4a3c3ba9f9743813d8a3ea46621b08abf965fd7181424d92a'
MANIFEST_SHA='c17b264fcdf34cf66980d087d729ddbe28d1f900859305f8d59f72f6f0ae8e51'
p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('--run',action='store_true');a=p.parse_args()
d=a.destination.resolve()
if d.exists() and any(d.iterdir()):raise SystemExit('destination must be empty')
b=base64.b64decode((ROOT/'CODE_CAPSULE.txt').read_text().strip(),validate=True)
if hashlib.sha256(b).hexdigest()!=CAPSULE_SHA:raise SystemExit('capsule hash mismatch')
payload=lzma.decompress(b)
if hashlib.sha256(payload).hexdigest()!=PAYLOAD_SHA:raise SystemExit('payload hash mismatch')
files=json.loads(payload)
if len(files)!=11:raise SystemExit('file count mismatch')
for name,text in files.items():
 q=(d/name).resolve()
 if not q.is_relative_to(d) or not isinstance(text,str):raise SystemExit('invalid archive member')
 q.parent.mkdir(parents=True,exist_ok=True);q.write_text(text,encoding='utf-8')
print('Restored',len(files),'source files to',d)
if a.run:
 (d/'results').mkdir(exist_ok=True)
 for script in ['run.py','followon.py','verify.py']:
  with (d/(script+'.log')).open('w') as out:
   subprocess.run([sys.executable,script],cwd=d,stdout=out,stderr=subprocess.STDOUT,check=True)
 mf=(d/'results/full_manifest.json').read_bytes()
 if hashlib.sha256(mf).hexdigest()!=MANIFEST_SHA:raise SystemExit('original scientific manifest mismatch')
 (d/'EXPECTED_MANIFEST.json').write_bytes(mf)
 subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=d,check=True)
 subprocess.run([sys.executable,'check_manifest.py'],cwd=d,check=True)
 print('Original 56-file scientific manifest reproduced; 22 tests passed.')
