"""Restore this round's recorded text capsule. Uses only the Python standard library.
Binary inputs/states and detailed traces are regenerated with the supplied program;
verify them with src/check_reproduction.py and the recorded32-file manifest.
"""
import argparse,base64,hashlib,json,lzma
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
meta=json.loads((root/'capsule_manifest.json').read_text())
blob=b''.join((root/name).read_bytes() for name in meta['parts'])
if hashlib.sha256(blob).hexdigest()!=meta['capsule_sha256']:raise SystemExit('capsule SHA256 mismatch')
obj=json.loads(lzma.decompress(base64.b64decode(blob,validate=False)))
if a.out.exists() and any(a.out.iterdir()):raise SystemExit('output must be empty or absent')
a.out.mkdir(parents=True,exist_ok=True)
for name,text in obj['files'].items():
    rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts:raise SystemExit('unsafe archive path')
    raw=text.encode('utf-8')
    if hashlib.sha256(raw).hexdigest()!=obj['sha256'][name]:raise SystemExit('file checksum mismatch: '+name)
    dst=a.out/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
print('Restored',len(obj['files']),'recorded text files')
