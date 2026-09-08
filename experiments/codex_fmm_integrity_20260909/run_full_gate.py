from pathlib import Path
import argparse,contextlib,hashlib,json,os,platform,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.exp_100a.run_experiment import execute,write_checksums
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
os.chdir(ROOT)
output=args.output_dir.resolve()
if output.exists():
    raise SystemExit('refuse overwrite')
output.mkdir(parents=True)
names=['experiments/exp_100a/run_experiment.py','experiments/exp_100a/config.json',
       'vortex_runtime/explicit_rectangular_fmm.py','experiments/codex_fmm_integrity_20260909/PREREGISTRATION.md',
       'experiments/codex_fmm_integrity_20260909/HARDENING_PREREGISTRATION.md',
       'experiments/codex_fmm_integrity_20260909/run_full_gate.py']
manifest={'sources':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},
 'invocation':[Path(sys.argv[0]).name]+sys.argv[1:],'python':sys.version,'numpy':np.__version__,
 'platform':platform.platform(),'processor':platform.processor(),
 'source_commit_field_is_base_revision':'executed working sources are separately hashed; no clean-base source claim',
 'GPU_executed':False}
with (output/'run.log').open('w',encoding='utf-8',newline='\n') as log,contextlib.redirect_stdout(log):
    result=execute(ROOT/'experiments/exp_100a/config.json',output)
for name,digest in manifest['sources'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
(output/'executed_source_manifest.json').write_bytes((json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode())
write_checksums(output)
print(json.dumps({'decision':result['authoritative_decision'],'integrity_failures':result['integrity_failures'],
 'shapes':len(result['search_rows']),'controls':result['catalog_summary'],
 'best_direct':result['best_direct_row'],'best_oracle':result['best_free_transform_oracle_row']}))
