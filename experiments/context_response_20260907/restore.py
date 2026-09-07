"""Restore an archival research capsule; verifies bytes, never executes its files."""
import argparse,base64,hashlib,json,lzma
from pathlib import Path,PurePosixPath
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
here=Path(__file__).resolve().parent
meta=json.loads((here/'capsule_meta.json').read_text())
b64=''.join((here/f).read_text().strip() for f in meta['parts'])
raw=base64.b64decode(b64,validate=True)
assert hashlib.sha256(raw).hexdigest()==meta['xz_sha256']
unpacked=lzma.decompress(raw)
assert hashlib.sha256(unpacked).hexdigest()==meta['json_sha256']
files=json.loads(unpacked)
assert set(files)==set(meta['file_sha256'])
if a.out.exists() and any(a.out.iterdir()):raise SystemExit('Output directory must be empty')
a.out.mkdir(parents=True,exist_ok=True)
for name,text in files.items():
 path=PurePosixPath(name)
 if path.is_absolute() or '..' in path.parts:raise SystemExit('Unsafe capsule path')
 data=text.encode('utf-8')
 assert hashlib.sha256(data).hexdigest()==meta['file_sha256'][name]
 dest=a.out.joinpath(*path.parts);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
print('Restored',len(files),'UTF-8 files. Binary evidence regenerates to expected manifest hashes.')
