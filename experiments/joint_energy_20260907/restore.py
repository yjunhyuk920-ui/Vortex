"""Restore the complete, recorded research artifacts; never executes restored code.

This capsule is archival compression, not an inference algorithm. Only standard
Python libraries are used. Existing differing files are never overwritten.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath

ARCHIVE_SHA256 = 'a910186456c5b5a0dca60d2ec3ef05e7a518a41055ede7520c1d9a8368da7c7e'
JSON_BYTES = 384314
FILE_COUNT = 78

def restore(destination: Path) -> dict[str, str]:
    here = Path(__file__).resolve().parent
    encoded = ''.join((here / f'part-{i:02}.b64').read_text().strip() for i in range(8))
    archive = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('archive checksum mismatch')
    raw = lzma.decompress(archive, memlimit=128 << 20)
    if len(raw) != JSON_BYTES:
        raise ValueError('decoded size mismatch')
    records = json.loads(raw)
    if len(records) != FILE_COUNT:
        raise ValueError('file count mismatch')
    root = destination.resolve()
    pending = []
    hashes = {}
    for name, mode, payload in records:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
            raise ValueError('unsafe path')
        if name in hashes:
            raise ValueError('duplicate path')
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError('path escapes destination')
        if mode == 'json':
            value = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode()
        elif mode == 'text':
            value = payload.encode('utf-8')
        elif mode == 'b64':
            value = base64.b64decode(payload, validate=True)
        else:
            raise ValueError('unknown record format')
        if target.exists() and target.read_bytes() != value:
            raise FileExistsError(f'refusing to overwrite {target}')
        hashes[name] = hashlib.sha256(value).hexdigest()
        pending.append((target, value))
    for target, value in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    return hashes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    hashes = restore(args.out)
    print(json.dumps({'restored_files': len(hashes), 'archive_sha256': ARCHIVE_SHA256,
                      'output': str(args.out.resolve())}, indent=2))
