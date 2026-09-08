"""Freeze existing evidence, then reproduce in a new directory without overwrites.

Freezing records the point at which this continuation observed existing results;
it does not retroactively make their original runs preregistered. The older
26-file manifest remains an independent immutable check.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def inventory(root,dirs):
    return {p.relative_to(root).as_posix():sha(p) for d in dirs
            for p in sorted((root/d).rglob('*')) if p.is_file()}

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--record',type=Path,default=HERE/'validation.json')
    a=p.parse_args();work=a.workdir.resolve()
    if HERE not in work.parents or work.exists():
        raise ValueError('new directory beneath experiment required')
    old=json.loads((HERE/'initial_results_manifest.json').read_text(encoding='utf-8'))
    for name,digest in old.items():
        if sha(HERE/'results'/name)!=digest:
            raise AssertionError('initial evidence changed: '+name)
    previous={q.relative_to(HERE/'results').as_posix():sha(q) for q in (HERE/'results').rglob('*') if q.is_file()}
    previous_path=HERE/'science_manifest.json'
    if previous_path.exists():
        if json.loads(previous_path.read_text(encoding='utf-8'))!=previous:
            raise AssertionError('previous full manifest mismatch')
    else:
        previous_path.write_text(json.dumps(previous,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    observed=inventory(HERE,['results','consumer_results'])
    frozen=HERE/'continuation_manifest.json'
    if frozen.exists():
        if json.loads(frozen.read_text(encoding='utf-8'))!=observed:
            raise AssertionError('fixed continuation manifest mismatch')
    else:
        frozen.write_text(json.dumps(observed,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    # Exact parent entry-point snapshots. No current documentation is overwritten.
    history=ROOT/'docs/research/history/pre_correlation_source_20260908'
    history.mkdir(parents=True,exist_ok=True)
    parent='66c01825635a357f86f9dfa5369088d7a239c92d'
    for name in ['README.md','RESEARCH_STATE.md','NEXT_EXPERIMENT.md','VALIDATION_MATRIX.md']:
        content=subprocess.check_output(['git','show',parent+':'+name],cwd=ROOT)
        dest=history/name
        if dest.exists() and dest.read_bytes()!=content:
            raise AssertionError('history mismatch')
        if not dest.exists():dest.write_bytes(content)
    work.mkdir(parents=True)
    files=['source.py','costs.py','test_source.py','verify.py','consumer_source.py',
           'test_consumer_source.py','verify_consumer.py']
    for name in files:shutil.copyfile(HERE/name,work/name)
    env=dict(os.environ,PYTHONUTF8='1');logs=[];test_count=None
    commands=[['-m','unittest','-v'],['source.py','--output','results'],
              ['costs.py','--output','results/costs.json'],
              ['verify.py','--results','results','--output','independent.json'],
              ['consumer_source.py','--output','consumer_results','--old-results','results'],
              ['verify_consumer.py','--results','consumer_results','--output','consumer_independent.json']]
    logdir=HERE/'logs'/work.name
    if logdir.exists():raise ValueError('new log directory required')
    logdir.mkdir(parents=True)
    for i,cmd in enumerate(commands):
        result=subprocess.run([sys.executable]+cmd,cwd=work,capture_output=True,text=True,
                              encoding='utf-8',env=env,timeout=90)
        logpath=logdir/f'command_{i}.txt'
        logpath.write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        logs.append(dict(command=cmd,exit_code=result.returncode,log=logpath.relative_to(HERE).as_posix()))
        if i==0:
            match=re.search(r'Ran (\d+) tests',result.stderr)
            if not match:raise AssertionError('test runner output missing')
            test_count=int(match.group(1))
        if result.returncode:raise RuntimeError('failure retained at '+str(logpath))
    fresh=inventory(work,['results','consumer_results'])
    if fresh!=observed:raise AssertionError('replay differs from frozen evidence')
    if json.loads((work/'independent.json').read_text())!=json.loads((HERE/'independent_validation.json').read_text()):
        raise AssertionError('prior independent validation changed')
    if json.loads((work/'consumer_independent.json').read_text())!=json.loads((HERE/'consumer_independent_validation.json').read_text()):
        raise AssertionError('consumer independent validation changed')
    record=dict(status='LOCAL_AUXILIARY_VALIDATION_PASS',test_count=test_count,
       deterministic_result_files=len(observed),initial_science_unchanged=len(old),
       initial_manifest_sha256=sha(HERE/'initial_results_manifest.json'),
       continuation_manifest_sha256=sha(frozen),sources={n:sha(HERE/n) for n in files},
       commands=logs,python=sys.version,platform=platform.platform(),
       theory_status='NOT_ESTABLISHED',hardware_status='NOT_TESTED',
       newly_closed_full_mission_obligations=0,
       not_tested=['public checkpoint','HF forward','full original RNG/KV','CUDA',
                   '405B','peak total8GiB','same-machine4BQ4 latency','TTFT','full repository tests'])
    if a.record.exists():raise ValueError('record exists; preserve it and choose another')
    a.record.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2,sort_keys=True))

if __name__=='__main__':main()
