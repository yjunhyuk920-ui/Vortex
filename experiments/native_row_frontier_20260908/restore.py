from pathlib import Path
import argparse,base64,hashlib,json,lzma,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
z=base64.b64decode((Path(__file__).parent/'CODE_CAPSULE.txt').read_text().strip(),validate=True)
assert hashlib.sha256(z).hexdigest()=='ef1eee9827cbd62bb8f48af958154b0f05c23f87107db66c2b8219e9c0b919e2'
raw=lzma.decompress(z,memlimit=134217728)
assert hashlib.sha256(raw).hexdigest()=='1e578851f98173c435b7b5fe83b2f394ead47da4a24f2eadc89f37cb660006f0'
files=json.loads(raw);assert len(files)==11
dest=Path(a.output).resolve()
if dest.exists() and any(dest.iterdir()):raise RuntimeError('Output must be empty')
dest.mkdir(parents=True,exist_ok=True)
for name,text in files.items():
 q=Path(name)
 if q.is_absolute() or '..' in q.parts:raise ValueError('Unsafe capsule path')
 out=dest/q;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text,encoding='utf-8')
print('Restored 11 checked files. Frozen science manifest is regenerated and hash-checked by verify.py.')
if a.run:subprocess.run([sys.executable,'verify.py'],cwd=dest,check=True)
