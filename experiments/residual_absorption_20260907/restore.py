"""Restore audited sources into an empty directory; optionally execute local replay.
Archive compression is transport, not a model compression or speed result.
No downloading/installing; needs the versions listed in README plus gcc.
"""
from pathlib import Path, PurePosixPath
import argparse,base64,hashlib,json,lzma,shutil,subprocess,sys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('destination');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    here=Path(__file__).resolve().parent;out=Path(a.destination).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('destination must be empty')
    m=json.loads((here/'CAPSULE_MANIFEST.json').read_text())
    z=base64.b64decode((here/'CODE_CAPSULE.txt').read_text().strip(),validate=True)
    if hashlib.sha256(z).hexdigest()!=m['xz_sha256']:raise ValueError('capsule hash mismatch')
    raw=lzma.decompress(z)
    if hashlib.sha256(raw).hexdigest()!=m['source_json_sha256']:raise ValueError('source JSON hash mismatch')
    d=json.loads(raw)
    if set(d)!=set(m['file_sha256']):raise ValueError('source inventory mismatch')
    for name,text in d.items():
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe archive path')
        if hashlib.sha256(text.encode()).hexdigest()!=m['file_sha256'][name]:raise ValueError('source hash mismatch')
        f=out/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(text)
    for name in ['PREREGISTRATION.md','FOLLOWON_PREREGISTRATION.md','LEDGER.md']:
        if (here/name).is_file():shutil.copy2(here/name,out/name)
    if (here/'REPORT.md').is_file():
        (out/'docs').mkdir(exist_ok=True);shutil.copy2(here/'REPORT.md',out/'docs/REPORT_KO.md')
    if a.run:
        for cmd in [[sys.executable,'run.py'],[sys.executable,'analyze.py'],[sys.executable,'-m','unittest','discover','-s','tests','-v'],[sys.executable,'verify.py']]:
            subprocess.run(cmd,cwd=out,check=True)
        v=json.loads((out/'results/validation.json').read_text())
        if v['manifest_sha256']!=m['replayed_run_manifest_sha256']:raise AssertionError('not original scientific result')
        if v['run_files_reproduced']!=m['expected_run_files']:raise AssertionError('wrong scientific inventory')
        print('SOURCE_HASHES_AND_ORIGINAL_RESULTS_VERIFIED')
    else:print('Sources restored; no runtime test claimed.')
if __name__=='__main__':main()
