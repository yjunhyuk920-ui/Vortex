"""Validate the evidence boundary without changing any frozen scientific byte."""
from pathlib import Path
import hashlib
import json
import subprocess
from verify import manifest

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    observed = ROOT / 'requirements.observed.txt'
    raw = observed.read_bytes()
    text = raw.decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    normalized = ROOT / 'requirements.replay.txt'
    normalized.write_bytes(text.encode('utf-8'))
    if observed.read_bytes() != raw:
        raise AssertionError('original environment observation was modified')
    frozen = json.loads((ROOT / 'results/frozen_manifest.json').read_text(encoding='utf-8-sig'))
    canonical = ROOT / 'results/run_v3'
    missing = []
    # The manifest uses canonical-science normalization for JSON, checked by verify.py.
    for name in frozen:
        if not (canonical / name).is_file():
            missing.append(name)
    if missing:
        raise AssertionError({'missing_canonical': missing})
    if manifest(canonical) != frozen:
        raise AssertionError('canonical scientific manifest changed')
    history = REPO/'docs/research/history/pre_native_global_transition_20260908'
    for name in ['README.md','RESEARCH_STATE.md','NEXT_EXPERIMENT.md','VALIDATION_MATRIX.md']:
        original = subprocess.check_output(['git','show','35e0a2eb1d9d64b20d17bd6e0c4afba07d3d70d3:'+name],cwd=REPO)
        saved = history/name
        if saved.read_bytes().replace(b'\r\n',b'\n') != original:
            raise AssertionError('history contains non-line-ending changes: '+name)
        saved.write_bytes(original)
    names = subprocess.check_output(['git','diff','--cached','--name-only','-z'], cwd=REPO).decode().split('\0')
    names = [p for p in names if p]
    forbidden = [p for p in names if '/.cache/' in p or '/.venv/' in p or '/results/replay_' in p or p.endswith('.safetensors')]
    if forbidden:
        raise AssertionError({'forbidden_staged': forbidden})
    binary_evidence = [p for p in names if '/native_global_transition_20260908/results/run_v3/' in p and p.endswith('.bin')]
    receipt = {
        'requirements_observed_sha256': sha(observed),
        'requirements_replay_sha256': sha(normalized),
        'normalization': 'BOM removed and CRLF converted to LF only in a separate replay file',
        'original_observation_preserved': True,
        'canonical_root': 'results/run_v3',
        'canonical_manifest_entries': len(frozen),
        'canonical_manifest_matches': True,
        'canonical_missing': missing,
        'forbidden_staged': forbidden,
        'staged_raw_output_binaries': len(binary_evidence),
        'staged_total_files_at_check': len(names),
        'full_repository_runtime_suite': 'NOT_TESTED',
    }
    dest = ROOT / 'results/persistence_boundary.json'
    dest.write_bytes((json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode())
    print(json.dumps(receipt,sort_keys=True))

if __name__ == '__main__':
    main()
