"""Restore the audited research text bundle; optional explicit local replay."""
from pathlib import Path, PurePosixPath
import argparse, base64, hashlib, json, lzma, subprocess, sys
HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('destination',type=Path);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    meta=json.loads((HERE/'CAPSULE_META.json').read_text())
    payload=''.join((HERE/name).read_text() for name in meta['chunks'])
    raw=base64.b64decode(payload,validate=True)
    if hashlib.sha256(raw).hexdigest()!=meta['xz_sha256']:raise SystemExit('xz checksum')
    decoded=lzma.decompress(raw)
    if hashlib.sha256(decoded).hexdigest()!=meta['text_json_sha256']:raise SystemExit('source checksum')
    files=json.loads(decoded)
    root=args.destination.resolve();root.mkdir(parents=True,exist_ok=True)
    if list(root.iterdir()):raise SystemExit('destination must be empty')
    for name,text in files.items():
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts:raise SystemExit('invalid archival path')
        out=root.joinpath(*p.parts);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text,encoding='utf-8')
    print('Restored',len(files),'text files to',root)
    if args.run:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=root,check=True)
        subprocess.run([sys.executable,'run.py'],cwd=root,check=True)
        manifest=root/'results/run/manifest.json'
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=meta['scientific_manifest_sha256']:raise SystemExit('regenerated evidence differs')
        subprocess.run([sys.executable,'verify.py'],cwd=root,check=True)
        print('Original scientific manifest matches after restoration')
if __name__=='__main__':main()
