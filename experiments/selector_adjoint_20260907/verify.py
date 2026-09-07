#!/usr/bin/env python3
"""Rerun from source and compare the pre-existing, frozen science manifest."""
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(__file__).resolve().parent
expected=(root/'EXPECTED_MANIFEST_SHA256.txt').read_text().strip()
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=root,check=True)
subprocess.run([sys.executable,'run.py'],cwd=root,check=True)
m=root/'results'/'run'/'manifest.json';actual=hashlib.sha256(m.read_bytes()).hexdigest()
if actual!=expected:raise SystemExit('Manifest mismatch: '+actual+' != '+expected)
manifest=json.loads(m.read_text())
for name,digest in manifest.items():
    if hashlib.sha256((m.parent/name).read_bytes()).hexdigest()!=digest:raise SystemExit('Content mismatch '+name)
print('PASS frozen science manifest',actual,'files',len(manifest))
