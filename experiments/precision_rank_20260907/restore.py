"""Restore checksummed scientific source, not a model-compression engine."""
from pathlib import Path, PurePosixPath
import argparse,base64,hashlib,json,lzma,subprocess,sys
XZ_SHA="0703b12fb7c9a985ac8afb0081aa4f9e064ab3b811eba67b647ab6d9c19e10dd"
JSON_SHA="98a7cc2fc5400a7c5c8c10be6fcf6e7580efa1aa220acb2824c08be2cf5c6706"
def main():
 p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--run',action='store_true');args=p.parse_args()
 root=Path(args.directory).resolve();root.mkdir(parents=True,exist_ok=True)
 raw=base64.b64decode((Path(__file__).parent/'CODE_CAPSULE.txt').read_text().strip(),validate=True)
 if hashlib.sha256(raw).hexdigest()!=XZ_SHA:raise RuntimeError('XZ checksum mismatch')
 data=lzma.decompress(raw)
 if hashlib.sha256(data).hexdigest()!=JSON_SHA:raise RuntimeError('source checksum mismatch')
 files=json.loads(data)
 for name,text in files.items():
  pos=PurePosixPath(name)
  if pos.is_absolute() or '..' in pos.parts:raise ValueError('unsafe archive path')
  dst=root/name
  if dst.exists() and dst.read_text()!=text:raise FileExistsError('refuse differing file: '+name)
  dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(text)
 print('Restored',len(files),'text files; raw science regenerates from them.')
 if args.run:subprocess.run([sys.executable,'verify.py'],cwd=root,check=True)
if __name__=='__main__':main()
