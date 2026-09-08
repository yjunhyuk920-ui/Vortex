from pathlib import Path
import argparse,collections,hashlib,json
from run_transition import compare_runs
ROOT=Path(__file__).resolve().parent

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):p.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n',encoding='utf-8')

def audit(run,previous=None):
    run=Path(run);s=load(run/'summary.json')
    assert s['weights_unchanged'] and not s['prefill_errors']
    assert all(not x['graph_errors'] and not x['packed_state_errors'] and not x['demand_errors'] for x in s['comparisons'])
    controls=[]
    if previous:
        for seed in [260908,260909,260910]:
            a=load(Path(previous)/f'reference_{seed}/run.json');b=load(run/f'reference_{seed}/run.json')
            errs=compare_runs(a,b);assert not errs
            controls.append({'seed':seed,'same_sequence':a['sequence']==b['sequence'],
                             'same_logits_kv_rng_layout':not errs})
    programs=[]
    for folder in ['programs','demand_programs']:
        for p in sorted((run/folder).glob('graph_*.json')):
            if p.name.endswith('_inventory.json'):continue
            d=load(p);i=load(p.with_name(p.stem+'_inventory.json'))
            attrs=d['attributes'];literals=[x for x in attrs.values() if x['kind']=='literal']
            checkpoint=[x for x in attrs.values() if x['kind']=='checkpoint']
            assert not d['model_weights_embedded']
            assert i['before']['matrix_macs']==i['after']['matrix_macs']
            assert i['before']['matrix_calls']==i['after']['matrix_calls']
            assert len(i['after']['root_names'])==61
            programs.append({'path':str(p.relative_to(run)).replace('\\','/'),
                'program_bytes':p.stat().st_size,'checkpoint_references':len(checkpoint),
                'literal_bytes':sum(len(x['hex'])//2 for x in literals),
                'largest_literal_bytes':max([len(x['hex'])//2 for x in literals] or [0]),
                'native_call_ratio':i['after']['calls']/i['before']['calls'],
                'matrix_mac_ratio':1.0,'root_count':len(i['after']['root_names']),
                'cse_operator_histogram':dict(collections.Counter(x['op'] for x in i['transform']['cse']))})
    matrix_shapes=[{'index':r['index'],'before_macs':r['before']['matrix_macs'],
                    'after_macs':r['after']['matrix_macs'],
                    'calls_before':r['before']['calls'],'calls_after':r['after']['calls'],
                    'logical_peak_bytes':r['after']['logical_last_use_peak_bytes']} for r in s['graph_programs']]
    result={'source_pin':s['source_pin'],'all_observed_logits_kv_rng_layout_match':True,
        'implicit_vs_explicit_mask_controls':controls,'programs':programs,'matrix_shapes':matrix_shapes,
        'constructor_native_forward_calls_A':len(s['graph_programs']),
        'constructor_native_forward_calls_C':s['demand_constructor_original_forward_calls'],
        'runtime_native_forward_calls_B':len(s['storage_records']),
        'codec_minimum_read_write_bytes_B':s['total_storage_codec_minimum_read_write_bytes'],
        'program_total_bytes':sum(p['program_bytes'] for p in programs),
        'original_checkpoint_file_bytes':269060552,'original_checkpoint_parameter_bytes':269030016,
        'frozen_weight_tensor_hash':s['weights_after'],
        'native_rhs_bytes_are_logical_operands_not_physical_traffic':True,
        'all_three_core_admission':False,'target_and_10x_not_established':True}
    save(run/'audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(ROOT/'results/run_v3'))
    p.add_argument('--previous',default=str(ROOT/'results/run_v2'));a=p.parse_args();audit(a.run,a.previous)
