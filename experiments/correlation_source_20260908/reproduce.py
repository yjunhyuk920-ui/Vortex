"""Fresh-source replay against a frozen manifest. No overwrites or downloads."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', type=Path, required=True)
    ap.add_argument('--record', type=Path)
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    work = args.workdir.resolve()
    if work.exists():
        raise ValueError('replay folder exists; choose a NEW folder')
    if here not in work.parents:
        raise ValueError('replay folder must be beneath this experiment')
    expected = json.loads((here/'science_manifest.json').read_text(encoding='utf-8'))
    work.mkdir(parents=True)
    for name in ['source.py', 'costs.py', 'test_source.py', 'verify.py']:
        shutil.copyfile(here/name, work/name)
    env = dict(os.environ, PYTHONUTF8='1')
    commands = [[sys.executable, '-m', 'unittest', '-v'],
                [sys.executable, 'source.py', '--output', 'results'],
                [sys.executable, 'costs.py', '--output', 'results/costs.json'],
                [sys.executable, 'verify.py', '--results', 'results', '--output', 'independent.json']]
    logs = []
    for i, cmd in enumerate(commands):
        p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True,
                           encoding='utf-8', timeout=90)
        (work/f'command_{i}.log').write_text(p.stdout+'\n'+p.stderr, encoding='utf-8')
        logs.append(dict(command=cmd[1:], returncode=p.returncode))
        if p.returncode:
            raise RuntimeError(f'command {i} failed; preserve logs')
    observed = {p.relative_to(work/'results').as_posix():digest(p)
                for p in (work/'results').rglob('*') if p.is_file()}
    if observed != expected:
        raise AssertionError('science differs from frozen manifest')
    assert json.loads((work/'independent.json').read_text()) == json.loads((here/'independent_validation.json').read_text())
    initial = json.loads((here/'initial_results_manifest.json').read_text())
    assert all(observed[n] == h for n, h in initial.items())
    obligations = json.loads((here/'obligations.json').read_text())
    assert obligations['newly_closed_full_mission_obligations'] == 0
    assert all(v['status'] == 'OPEN' for v in obligations['obligations'])
    data = dict(status='LOCAL_AUXILIARY_VALIDATION_PASS', test_count=11,
                deterministic_result_files=len(observed), frozen_manifest_sha256=digest(here/'science_manifest.json'),
                initial_science_files_unchanged=len(initial), commands=logs,
                independent=json.loads((work/'independent.json').read_text()),
                python=sys.version, platform=platform.platform(),
                theory_status='NOT_ESTABLISHED', hardware_status='NOT_TESTED',
                closed_full_mission_obligations=0, full_repository_tests='NOT TESTED')
    text = json.dumps(data, indent=2, sort_keys=True)+'\n'
    if args.record:
        args.record.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
