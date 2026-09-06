"""Restore losslessly recorded observations. No network or model execution."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json

def unpack(source: Path, target: Path) -> None:
    bundle=json.loads(gzip.decompress(base64.b64decode(source.read_text(),validate=False)))
    target.mkdir(parents=True,exist_ok=True)
    for name,item in bundle.items():
        if Path(name).name != name:
            raise ValueError('unsafe recorded file name')
        data=item['utf8'].encode('utf-8')
        if hashlib.sha256(data).hexdigest()!=item['sha256']:
            raise ValueError('recorded SHA256 mismatch: '+name)
        (target/name).write_bytes(data)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('target',type=Path)
    a=p.parse_args();unpack(a.source,a.target)
