"""Restore a checksummed source-only archive; --run explicitly runs validation."""
from pathlib import Path, PurePosixPath
import argparse,base64,hashlib,json,lzma,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('destination');p.add_argument('--run',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parent
meta=json.loads((root/'CAPSULE_META.json').read_text())
packed=base64.b64decode(''.join(''.join((root/name).read_text() for name in meta['parts']).split()),validate=True)
if hashlib.sha256(packed).hexdigest()!=meta['xz_sha256']:raise SystemExit('XZ checksum mismatch')
raw=lzma.decompress(packed)
if hashlib.sha256(raw).hexdigest()!=meta['source_json_sha256']:raise SystemExit('Source checksum mismatch')
files=json.loads(raw);target=Path(a.destination).resolve();target.mkdir(parents=True,exist_ok=True)
if len(files)!=meta['file_count']:raise SystemExit('File count mismatch')
for name,text in files.items():
    q=PurePosixPath(name)
    if q.is_absolute() or '..' in q.parts:raise SystemExit('Unsafe source path')
    dest=target.joinpath(*q.parts)
    if not dest.resolve().is_relative_to(target):raise SystemExit('Path escapes destination')
    if dest.exists() and dest.read_text()!=text:raise SystemExit('Refusing to overwrite differing file: '+name)
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
print('Restored',len(files),'verified text files to',target)
if a.run:subprocess.run([sys.executable,'verify.py'],cwd=target,check=True)
