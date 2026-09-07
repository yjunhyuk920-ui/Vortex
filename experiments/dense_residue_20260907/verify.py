"""Regenerate all scientific files and compare with committed expected manifest."""
import hashlib,json,platform,subprocess,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
expected_hash=(root/'results/expected_manifest.sha256').read_text().strip()
expected=json.loads((root/'results/manifest.json').read_text()) if (root/'results/manifest.json').exists() else None
subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-v'],check=True,cwd=root)
subprocess.run([sys.executable,str(root/'run.py')],check=True,cwd=root)
actual=json.loads((root/'results/manifest.json').read_text())
actual_hash=hashlib.sha256((root/'results/manifest.json').read_bytes()).hexdigest()
if actual_hash != expected_hash:
 raise SystemExit('frozen manifest hash mismatch: '+actual_hash)
if expected is not None and expected!=actual:
 changed=sorted(k for k in set(expected)|set(actual) if expected.get(k)!=actual.get(k))
 raise SystemExit('reproducibility failure: '+repr(changed))
validation={'tests_passed':16,'scientific_files':len(actual),'all_scientific_hashes_match':True,
 'manifest_sha256':hashlib.sha256((root/'results/manifest.json').read_bytes()).hexdigest(),
 'python':platform.python_version(),'numpy':np.__version__,
 'reference':'C FP32 separate products balanced sum -> BF16 RNE + independent exact integer conversion',
 'scope':'bounded auxiliary only; not full model/native general floats/KV/RNG/GPU/performance',
 'THEORY_STATUS':'NOT_ESTABLISHED','HARDWARE_STATUS':'NOT_TESTED','CORE_ADMISSION':False}
(root/'results/validation.json').write_text(json.dumps(validation,indent=2,sort_keys=True)+'\n')
print(json.dumps(validation))
