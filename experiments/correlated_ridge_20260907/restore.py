"""Restore archived research text; this does not run the restored code."""
import argparse
import base64
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath

EXPECTED = '7d7bb45c2441470153969cd2abf863382d47bd74864a32ecff737a0a50b02494'

def restore(source: Path, out: Path) -> None:
    parts = [source / f'capsule.part{i:02}.b64' for i in range(1, 8)]
    text = ''.join(p.read_text(encoding='utf-8').strip() for p in parts)
    raw = lzma.decompress(base64.b64decode(text, validate=True))
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('Archive digest mismatch')
    files = json.loads(raw)
    if not isinstance(files, dict) or len(files) != 25:
        raise ValueError('Invalid archive manifest')
    out = out.resolve()
    for name, content in files.items():
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or not isinstance(content, str):
            raise ValueError('Invalid archive path or content')
        target = (out / Path(*p.parts)).resolve()
        if not target.is_relative_to(out):
            raise ValueError('Path leaves output directory')
        if target.exists() and target.read_bytes() != content.encode('utf-8'):
            raise FileExistsError(f'Refusing to replace different data: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode('utf-8'))
    print(f'Restored {len(files)} text files into {out}. No code was executed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    restore(Path(__file__).resolve().parent, args.out)
