"""Restore checked local research sources, never execute them automatically."""
from pathlib import Path
import argparse, base64, hashlib, json, lzma
ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, default=ROOT/'restored')
a = p.parse_args()
text = ''.join((ROOT/f'CAPSULE_{i}.txt').read_text() for i in range(5))
packed = base64.b64decode(text, validate=False)
assert hashlib.sha256(packed).hexdigest() == '81050105dcbfba723defdb8c9ee05901d7777cd21de2704c59a4cd3c99363243'
raw = lzma.decompress(packed)
assert hashlib.sha256(raw).hexdigest() == '6c737b39b9e8e984558d916b00dd832cfe1104e42420e1b3ad6b246e476802d9'
files = json.loads(raw)
assert len(files) == 19
root = a.output.resolve()
for name, content in files.items():
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('unsafe archive path')
    dest = (root/rel).resolve()
    if root not in dest.parents:
        raise ValueError('unsafe archive target')
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode('utf-8')
    if dest.exists() and dest.read_bytes() != data:
        raise FileExistsError(f'refuse to overwrite changed file: {dest}')
    dest.write_bytes(data)
print(f'Restored {len(files)} verified text files to {root}')
print('Review docs/REPORT_KO.md; run python verify.py in that directory to replay.')
