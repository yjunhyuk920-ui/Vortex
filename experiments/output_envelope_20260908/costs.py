"""Post-result accounting; no parameter search and no target latency prediction."""
from pathlib import Path
from fractions import Fraction
import json

ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'results/science/summary.json').read_text(encoding='utf-8'))
raw=s['raw_tensor_bytes_total']; meta=s['summary_payload_bytes_total']
cost={
    'kind':'DERIVED_FROM_FROZEN_FORMAT_NOT_WALL_CLOCK',
    'public_raw_tensor_bytes':raw,
    'metadata_payload_bytes':meta,
    'original_storage_retained':True,
    'public_runtime_weight_plus_envelope_payload_ratio':float(Fraction(raw+meta,raw)),
    'public_minimum_build_then_8_queries_ratio':float(Fraction(raw+meta+8*(raw+meta),8*raw)),
    'previous_ratio_includes':['one_original_constructor_read','summary_payload_write','8_original_reads','8_summary_reads'],
    'previous_ratio_excludes':['initial_network_acquisition','summary_reload','header_bytes','constructor_extrema_comparisons','input_decode','output_encode','allocation','temporary_array_read_write','file_open_seek','cache_effects','independent_reference_validation'],
    'general_equation':{'groups':'g=ceil(m/b)','certified_rows':'c',
                        'runtime_coefficient_payload':'4*g*n+2*(m-c)*n',
                        'separate_FP_arithmetic':'(2*g+m-c)*(n+nextpow2(n)-1)',
                        'constructor_work_upper_order':'O(m*n) value scans/comparisons, O(g*n) output coefficients',
                        'current_constructor_memory_order':'O(m*n) host arrays, not an 8GiB streaming implementation',
                        'runtime_live_memory_order':'O(g*n+b*nextpow2(n)+m+n) arrays plus original file/OS caches'},
    'uniform_packet_256':{'ratio':'2/256+1-f',
                          'necessary_certified_row_fraction_for_10x':float(Fraction(9,10)+Fraction(2,256)),
                          'requires_all_other_costs_too':True},
    'synthetic_405e9_coefficient_illustration':{
        'MEASURED':False,'all_matrices_assumed_full_packets':True,
        'original_BF16_bytes':810000000000,
        'envelope_payload_bytes':6328125000,
        'envelope_payload_GiB':6328125000/2**30,
        'not_a_405B_compressed_model':True,
        'required_extra':['original weights','KV','code','metadata','buffers','allocator','certificate work']},
    'full_mission_O1_O6':'OPEN', 'target_latency':'NOT_TESTED',
    'gate_decision':'REJECT_FIXED_PACKET_SHARED_OUTPUT_CORE_RETAIN_SCOPED_REFERENCE'
}
(ROOT/'results/costs.json').write_text(json.dumps(cost,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps(cost,ensure_ascii=False,sort_keys=True))
