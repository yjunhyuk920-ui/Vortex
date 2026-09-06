from pathlib import Path
import argparse,base64,hashlib,json,lzma

def restore(out):
    here=Path(__file__).resolve().parent
    meta=json.loads((here/'CAPSULE.json').read_text())
    text=''.join((here/f'capsule-{i:02d}.b64').read_text().strip() for i in range(meta['parts']))
    packed=base64.b64decode(text,validate=True)
    if hashlib.sha256(packed).hexdigest()!=meta['xz_sha256']: raise ValueError('capsule hash')
    raw=lzma.decompress(packed)
    if hashlib.sha256(raw).hexdigest()!=meta['text_sha256']: raise ValueError('text hash')
    files=json.loads(raw); out=out.resolve()
    if len(files)!=meta['files']: raise ValueError('file count')
    for name,content in files.items():
        p=(out/name).resolve()
        if not p.is_relative_to(out): raise ValueError('unsafe path')
        if p.exists() and p.read_text()!=content: raise FileExistsError(p)
        p.parent.mkdir(parents=True,exist_ok=True); p.write_text(content)
    print(f'Restored {len(files)} text files to {out}')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    restore(ap.parse_args().out)
