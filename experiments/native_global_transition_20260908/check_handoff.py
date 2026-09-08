"""Validate changed documentation, scope, source and frozen science before Git.

Does not rewrite any frozen numerical hashes or include downloaded weights.
"""
from pathlib import Path
import hashlib,json,re,subprocess
from verify import manifest
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
BASE='35e0a2eb1d9d64b20d17bd6e0c4afba07d3d70d3'
ENTRY=['README.md','RESEARCH_STATE.md','NEXT_EXPERIMENT.md','VALIDATION_MATRIX.md']
LEDGER=['ARCHITECTURE.md','ASSUMPTION_REGISTER.md','DECISION_LOG.md',
        'HARDWARE_VALIDATION_PLAN.md','REPRODUCIBILITY.md','VALIDATION.md']
POLICY=['AGENTS.md','MISSION_AND_WORKING_PRINCIPLES.md','CREATIVE_RESEARCH_MANDATE.md',
        'docs/CONSTRUCTIVE_THEORY_CONTRACT.md','docs/PROOF_FIRST_CONTRACT.md',
        'docs/RESEARCH_EFFICIENCY_CONTRACT.md','docs/WORK_SESSION_PROTOCOL.md',
        'docs/REPOSITORY_COMMIT_AND_HANDOFF_MANDATE.md']

def original(path):
    return subprocess.check_output(['git','show',f'{BASE}:{path}'],cwd=REPO)

def main():
    errors=[];hist={};pol={};links=0
    for f in ENTRY:
        old=original(f);saved=(REPO/'docs/research/history/pre_native_global_transition_20260908'/f).read_bytes()
        if old!=saved:errors.append(f'history differs:{f}')
        hist[f]=hashlib.sha256(old).hexdigest()
    for f in POLICY:
        old=original(f)
        expected_blob=subprocess.check_output(['git','rev-parse',f'{BASE}:{f}'],cwd=REPO).strip()
        clean_blob=subprocess.check_output(['git','hash-object','--path='+f,f],cwd=REPO).strip()
        if expected_blob!=clean_blob:errors.append(f'policy changed:{f}')
        pol[f]=hashlib.sha256(old).hexdigest()
    paths=[REPO/f for f in ENTRY+LEDGER]+list(ROOT.glob('*.md'))
    for p in paths:
        text=p.read_text(encoding='utf-8')
        if text.count('```')%2:errors.append(f'unpaired fence:{p.name}')
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
            if '://' in target or target.startswith('#'):continue
            q=(p.parent/target.split('#')[0]).resolve();links+=1
            if not q.exists():errors.append(f'broken link:{p.name}:{target}')
    frozen=json.loads((ROOT/'results/frozen_manifest.json').read_text())
    if manifest(ROOT/'results/run_v3')!=frozen:errors.append('science changed after freeze')
    s=json.loads((ROOT/'results/run_v3/summary.json').read_text())
    a=json.loads((ROOT/'results/run_v3/audit.json').read_text())
    if not s['weights_unchanged']:errors.append('weights changed')
    for v in a['programs']:
        if v['matrix_mac_ratio']!=1.0 or v['root_count']!=61:errors.append('incorrect report matrix/root claim')
    if a['codec_minimum_read_write_bytes_B']!=5114880:errors.append('codec cost claim')
    if s['reference_generation_logit_coordinates']!=589824 or s['reference_kv_coordinates']!=760320:
        errors.append('coordinate count claim')
    report={'base':BASE,'errors':errors,'relative_links_checked':links,
        'policy_unchanged_sha256':pol,'policy_comparison':'unchanged Git-clean blob identities',
        'previous_entry_sha256':hist,'history_comparison':'literal original Git blob bytes, no normalization',
        'frozen_science_count':len(frozen),'matrix_cost_and_root_claims_checked':True,
        'whole_repository_runtime_suite':'NOT_TESTED','target_hardware':'NOT_TESTED'}
    (ROOT/'results/handoff_validation.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2));assert not errors

if __name__=='__main__':main()
