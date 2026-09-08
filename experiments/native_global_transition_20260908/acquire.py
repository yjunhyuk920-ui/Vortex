"""Download only explicitly pinned public model files; verify upstream LFS hash."""
from pathlib import Path
import hashlib,json,time,requests
ROOT=Path(__file__).resolve().parent
MODEL='HuggingFaceTB/SmolLM2-135M'
PIN='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
FILES=['config.json','generation_config.json','model.safetensors']

def hfile(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def acquire():
    cache=ROOT/'.cache/model';cache.mkdir(parents=True,exist_ok=True)
    result=ROOT/'results';result.mkdir(exist_ok=True)
    response=requests.get(f'https://huggingface.co/api/models/{MODEL}/tree/{PIN}',params={'recursive':'true','expand':'true'},timeout=(15,90))
    response.raise_for_status();items=response.json()
    if not isinstance(items,list):raise RuntimeError('invalid repository tree')
    records=[]
    for name in FILES:
        entries=[v for v in items if v.get('path')==name]
        if len(entries)!=1:raise RuntimeError(('expected file absent or repeated',name))
        item=entries[0];path=cache/name;start=time.perf_counter();downloaded=False
        if not path.exists():
            tmp=cache/(name+'.partial')
            if tmp.exists():raise RuntimeError(('prior partial preserved; inspect before retry',str(tmp)))
            r=requests.get(f'https://huggingface.co/{MODEL}/resolve/{PIN}/{name}',stream=True,timeout=(15,120))
            r.raise_for_status()
            count=0
            with tmp.open('xb') as f:
                for part in r.iter_content(1<<20):
                    count+=len(part)
                    if count>int(item['size']):raise RuntimeError('payload exceeds pinned size')
                    f.write(part)
            if count!=int(item['size']):raise RuntimeError('truncated model payload')
            tmp.rename(path);downloaded=True
        actual=hfile(path);expected=item.get('lfs',{}).get('oid')
        if path.stat().st_size!=item['size']:raise RuntimeError('cached size mismatch')
        if expected and actual!=expected:raise RuntimeError(('cached SHA256 mismatch',name,actual,expected))
        records.append({'path':name,'bytes':path.stat().st_size,'sha256':actual,
                        'upstream_lfs_sha256':expected,'pinned_git_blob':item.get('oid'),
                        'downloaded_now':downloaded,'read_download_verify_seconds':time.perf_counter()-start})
        print(json.dumps(records[-1]),flush=True)
    manifest={'model':MODEL,'revision':PIN,'files':records,'weight_modified':False,'trust_remote_code':False}
    out=result/'acquisition.json'
    if out.exists():
        old=json.loads(out.read_text(encoding='utf-8'))
        if [(v['path'],v['sha256']) for v in old['files']]!=[(v['path'],v['sha256']) for v in records]:
            raise RuntimeError('reacquisition differs from frozen manifest')
    else:out.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    return cache

if __name__=='__main__':print(acquire())
