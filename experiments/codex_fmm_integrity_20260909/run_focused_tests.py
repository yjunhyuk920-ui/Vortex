from pathlib import Path
import argparse,hashlib,importlib.util,inspect,json,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests/exp_100a'))
import test_explicit_rectangular_fmm as legacy
import test_search_integrity as focused
passed=[]
for name,value in sorted(vars(legacy).items()):
    if not name.startswith('test_') or not inspect.isfunction(value):
        continue
    parameters=list(inspect.signature(value).parameters)
    if not parameters:
        value()
    elif parameters==['tmp_path']:
        with tempfile.TemporaryDirectory(prefix='fmm-test-') as temp:
            value(Path(temp))
    else:
        raise RuntimeError('unsupported test fixture '+name)
    passed.append(name)
suite=unittest.defaultTestLoader.loadTestsFromModule(focused)
run=unittest.TextTestRunner(verbosity=2).run(suite)
if not run.wasSuccessful():
    raise SystemExit(1)
result={'legacy_test_functions_passed':passed,'legacy_count':len(passed),'new_unittest_count':run.testsRun,
 'method':'existing plain test functions directly invoked, explicit TemporaryDirectory for tmp_path; new tests via unittest; pytest unavailable in bundled environment',
 'runner_sha256':hashlib.sha256((ROOT/'experiments/exp_100a/run_experiment.py').read_bytes()).hexdigest()}
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path(__file__).parent/'focused_validation.json')
path=parser.parse_args().output
if path.exists():
    raise SystemExit('refuse overwrite of focused evidence')
path.write_bytes((json.dumps(result,indent=2,sort_keys=True)+'\n').encode())
print(json.dumps(result))
