"""Record actual pushed scientific commit; receipt never names its own future SHA."""
from pathlib import Path
import hashlib
import json
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
EXPECTED = '59a5e106a1475fbe4eaa649bd2a392c989335286'
BRANCH = 'research/native-global-transition-20260908'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO).decode().strip()

def main():
    dest = ROOT / 'results/remote_verified_v1.json'
    if dest.exists():
        raise RuntimeError('Existing receipt is immutable')
    head = git('rev-parse', 'HEAD')
    remote = git('ls-remote', 'origin', 'refs/heads/' + BRANCH).split()[0]
    if head != EXPECTED or remote != EXPECTED:
        raise AssertionError({'expected': EXPECTED, 'head': head, 'remote': remote})
    names = ['README.md', 'RESEARCH_STATE.md', 'NEXT_EXPERIMENT.md',
             'experiments/native_global_transition_20260908/REPORT.md',
             'experiments/native_global_transition_20260908/native_graph.py',
             'experiments/native_global_transition_20260908/native_state.py',
             'experiments/native_global_transition_20260908/verify.py',
             'experiments/native_global_transition_20260908/results/validation_final.json']
    blobs = {}
    for name in names:
        content = subprocess.check_output(['git','show',EXPECTED+':'+name],cwd=REPO)
        blobs[name] = {'git_blob': git('rev-parse', EXPECTED+':'+name),
                       'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    receipt = {
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'repository': 'yjunhyuk920-ui/Vortex', 'branch': BRANCH,
        'scientific_commit': EXPECTED, 'local_head': head, 'remote_ref': remote,
        'draft_pr': 149, 'remote_commit_verified': True,
        'git_content_address_checks': blobs,
        'connector_readbacks': ['PR149 head_sha', 'REST branch ref',
            'experiment directory at scientific commit', 'native_graph.py contents', 'raw REPORT.md'],
        'scientific_status': 'NOT_ESTABLISHED', 'target_hardware': 'NOT_TESTED',
        'receipt_own_future_commit': None,
    }
    with dest.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'remote_commit_verified':True,'scientific_commit':head,'files_checked':len(blobs)}))

if __name__ == '__main__':
    main()
