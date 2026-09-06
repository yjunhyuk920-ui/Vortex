"""Lossless evidence packaging; raw output equality is encoded, never assumed."""
from __future__ import annotations
import argparse,gzip,hashlib,json,re
from pathlib import Path

def readj(p):return json.loads(p.read_text())
def json_bytes(x):return (json.dumps(x,sort_keys=True,indent=2)+'\n').encode()

def pack(folder:Path,dest:Path):
    names=['inputs.json','frozen.json','summary.json','measurements.json','validation.json']
    obj={'version':1,'files':{n:readj(folder/n) for n in names},'tags':[],'rows':[],
         'file_sha256':{n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in names}}
    tags={}
    for line in (folder/'raw.jsonl').read_text().splitlines():
        r=json.loads(line);t=r['tag']
        if t not in tags:tags[t]=len(tags);obj['tags'].append(t)
        c=r['counters'];y=r['y']
        obj['rows'].append([tags[t],r['query_index'],y,
                           None if r['native']==y else r['native'],
                           None if r['integer_ref']==y else r['integer_ref'],
                           c['visited_nodes'],c['memo_lookups'],c['input_bit_tests'],
                           c['unique_64byte_blocks'],c['unique_4096byte_pages']])
    obj['file_sha256']['raw.jsonl']=hashlib.sha256((folder/'raw.jsonl').read_bytes()).hexdigest()
    data=json.dumps(obj,sort_keys=True,separators=(',',':')).encode()
    dest.write_bytes(gzip.compress(data,mtime=0))

def unpack(source:Path,out:Path):
    packed=(b''.join((source/f'evidence.part{i:02d}').read_bytes() for i in range(5))
            if source.is_dir() else source.read_bytes())
    data=gzip.decompress(packed)
    if len(data)>64_000_000:raise ValueError('evidence size limit')
    obj=json.loads(data)
    if obj['version']!=1:raise ValueError('version')
    out.mkdir(parents=True,exist_ok=True)
    names=['inputs.json','frozen.json','summary.json','measurements.json','validation.json']
    for name in names:(out/name).write_bytes(json_bytes(obj['files'][name]))
    cases={(c['case_id'],c['seed']):c for c in obj['files']['inputs.json']}
    with (out/'raw.jsonl').open('w') as f:
        for idx,qi,y,ny,iy,v,lookups,t,b64,p4096 in obj['rows']:
            tag=obj['tags'][idx];match=re.fullmatch(r'c(\d+)_s(\d+)_(input|plane)',tag)
            if not match:raise ValueError('tag')
            case=cases[(int(match[1]),int(match[2]))];m,n=case['m'],case['n']
            counters=dict(visited_nodes=v,node_payload_bytes=12*v,root_payload_bytes=64*m,
                          varmap_bytes=4*t,model_code_bytes=12*v+64*m+4*t,memo_lookups=lookups,
                          memo_inserts=v,input_bit_tests=t,memo_value_bytes_lower_bound=5*v,
                          unique_64byte_blocks=b64,unique_4096byte_pages=p4096,input_bytes=2*n,output_bytes=2*m)
            r=dict(tag=tag,query_index=qi,x=case['queries'][qi],y=y,native=y if ny is None else ny,
                   integer_ref=y if iy is None else iy,counters=counters)
            f.write(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n')
    for name,sha in obj['file_sha256'].items():
        if name not in names+['raw.jsonl']:raise ValueError('filename')
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=sha:raise ValueError('hash mismatch '+name)
    return obj['file_sha256']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pack','unpack']);p.add_argument('source');p.add_argument('dest');a=p.parse_args()
    if a.mode=='pack':pack(Path(a.source),Path(a.dest))
    else:print(json.dumps(unpack(Path(a.source),Path(a.dest)),indent=2))
