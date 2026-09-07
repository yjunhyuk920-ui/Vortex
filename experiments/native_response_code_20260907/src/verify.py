"""Check the exact manifest of the 171 generated experiment files."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
a=ROOT/'results/run'
manifest={str(p.relative_to(a)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.rglob('*')) if p.is_file()}
payload=json.dumps(manifest,indent=2,sort_keys=True)+'\n'
expected=json.loads((ROOT/'results/validation.json').read_text())
assert len(manifest)==expected['reproducible_file_count'],len(manifest)
assert hashlib.sha256(payload.encode()).hexdigest()==expected['generated_manifest_sha256'],'manifest mismatch; check pinned ABI/compiler'
print('Verified',len(manifest),'generated files against the recorded full manifest hash.')
