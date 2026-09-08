"""Validate persisted evidence and record handoff facts without inventing results."""
from pathlib import Path
import hashlib,json,subprocess,sys,datetime,re
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
required=['docs/REPORT_KO.md','OBLIGATIONS.md','HANDOFF.md','README.md','PREREGISTRATION.md','results/costs.json','results/science/summary.json','results/science/manifest.json','logs/verify_replay.txt','logs/portable_replay.txt']
for name in required:
    assert (ROOT/name).is_file(),name
expected=json.loads((ROOT/'results/science/manifest.json').read_text(encoding='utf-8'))
for name,h in expected.items():assert digest(ROOT/'results/science'/name)==h,name
for path in ['replay/verified','replay/portable_check']:
    for name,h in expected.items():
        if name!='source_before_payload.json':assert digest(ROOT/path/name)==h,(path,name)
s=json.loads((ROOT/'results/science/summary.json').read_text(encoding='utf-8'))
assert (s['matrices'],s['queries'],s['output_coordinates'])==(12,96,101376)
assert s['accepted_groups']==s['oracle_broadcastable_groups']==s['mismatches']==0
initial=json.loads((ROOT/'results/science/source_before_payload.json').read_text(encoding='utf-8'))
for name,h in initial.items():assert digest(ROOT/name)==h,('initial source altered',name)
history={}
for name in ['README.md','RESEARCH_STATE.md','NEXT_EXPERIMENT.md','VALIDATION_MATRIX.md']:
    canonical=subprocess.check_output(['git','show','66c01825635a357f86f9dfa5369088d7a239c92d:'+name],cwd=REPO)
    archived=(REPO/'docs/research/history/pre_output_envelope_20260908'/name).read_bytes()
    assert canonical==archived.replace(b'\r\n',b'\n'),('prior content differs',name)
    history[name]={'canonical_sha256':hashlib.sha256(canonical).hexdigest(),'local_archive_sha256':hashlib.sha256(archived).hexdigest(),'git_text_normalization_checked':True}
checks=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(ROOT/'logs/unit_final.txt').write_text(checks.stdout,encoding='utf-8')
if checks.returncode:raise RuntimeError(checks.stdout)
links=[]
for name in ['README.md','RESEARCH_STATE.md','NEXT_EXPERIMENT.md','VALIDATION_MATRIX.md']:
    text=(REPO/name).read_text(encoding='utf-8')
    for url in re.findall(r'\]\(([^)]+)\)',text):
        if '://' not in url and not url.startswith('#'):
            assert (REPO/url.split('#')[0]).exists(),(name,url)
            links.append([name,url])
result={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'PASS','unit_tests':12,'initial_scientific_files_checked':len(expected),
        'initial_source_hashes_unchanged':True,'fresh_replays':2,
        'numerical_files_compared_per_replay':len(expected)-1,
        'required_files_present':required,'prior_entrypoint_preservation':history,
        'entrypoint_relative_links_checked':links,
        'HF_FORWARD':'NOT_TESTED','FULL_KV_RNG':'NOT_TESTED','TARGET_405B':'NOT_TESTED',
        'THEORY_STATUS':'NOT_ESTABLISHED','CORE_ADMISSION':False,'full_repository_suite':'NOT_TESTED'}
(ROOT/'results/validation.json').write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,sort_keys=True))
