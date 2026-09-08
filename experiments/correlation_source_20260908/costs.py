"""Explicit file-layout bounds and paid finite algorithms, not latency claims."""
from __future__ import annotations
import json
from pathlib import Path


def layout(n: int, k: int, r: int) -> dict:
    if not (n >= 1 and k >= 1 and 1 <= r <= min(n, 1 << k)):
        raise ValueError('inconsistent dimensions')
    width = (r - 1).bit_length()
    pattern_bytes = r * ((k + 7) // 8)
    id_bytes = (n * width + 7) // 8
    # Conceptual payloads can be split into legal files; this does not allocate.
    return dict(n=n, k=k, distinct_patterns=r, id_bits=width,
                original_packed_bytes=(n*k+7)//8,
                dictionary_bytes=pattern_bytes, positional_id_bytes=id_bytes,
                single_header_and_digest_bytes=64,
                conceptual_one_header_bytes=64+pattern_bytes+id_bytes,
                scale_only=True)


def run(path: Path) -> dict:
    a = layout(405_000_000_000, 16, 65_536)
    a['format_warning'] = 'Must shard uint32 dictionary/dimensions if exceeded. Count every shard header. Not an allocated file or measured alphabet.'
    a['positional_id_GiB'] = a['positional_id_bytes'] / 2**30
    data = dict(
        formula='D=64+r*ceil(k/8)+ceil(N*ceil(log2(r))/8)',
        scan='Q=popcount(rowmask)*popcount(colmask); exactly Q ID lookups and histogram increments, even when every function cancels',
        histogram_counter_bits='ceil(log2(N+1)) per slot suffices in a packed-word implementation; Python object overhead is additional',
        source_residency='Current Source keeps the COMPLETE serialized byte string in host RAM. No SSD/page cache or GPU placement implemented.',
        source_read_upper='Q*ceil((ceil(log2(r))+7)/8)+min(r,Q)*ceil(k/8), plus masks, histogram reads/writes, outputs and program; width=0 gives zero ID bytes',
        constructor='Read Nk source bits; comparison-sort N k-bit words; N binary searches in r patterns; write r patterns and N IDs; hash complete file',
        logical_sort_model='Balanced mergesort would require <=N*ceil(log2(N)) comparisons; Python sorted is used and not presented as an exact counted mergesort run. O(N log N) comparison bound only.',
        complete_query_stages=['validate query/function domain', 'allocate/zero r counters and F outputs',
            'scan row/column masks', 'read selected positional IDs', 'read/increment histogram counts',
            'scan r counters', 'read patterns with odd counts', 'evaluate every supplied ANF monomial',
            'write F parity outputs', 'release temporary histogram'],
        bit_cost_upper='O(N log(N)*(k+log(N))+Nk + r log(N) + m^2 + m*n^2 + Q*(log(N)+k) + r + min(r,Q)*L*k + F); includes big-int shifts pessimistically; not a hardware time in seconds',
        bit_space_upper='O(D*8+Nk+N log(N)+r log(N)+L*k+F+m+n); original input and constructor copies included; interpreter/object constants and allocator not numerically bounded',
        native_order_continuation='All positional IDs and pattern words are read; a padded balanced tree with P=next_power_of_two(N) uses P-1 native adds. Producing FP32 product words from W,x is NOT removed.',
        finite_boolean_relation='For elimination buckets j with union scope s_j and d_j factors, direct Boolean factor enumeration costs <=sum_j (d_j+1)*2^s_j logical table steps, plus factor construction and all I/O; no small-scope guarantee supplied.',
        transformed_state='Full table construction over B-bit states and A legal actions costs A*2^B native transitions before partition refinement. A cheap nonenumerative native conjugacy is NOT CONSTRUCTED.',
        full_word_layout=a,
        target_timing='NOT TESTED', full_peak_gpu='NOT TESTED',
        same_machine_4b_lower_or_coupled_bound='NOT ESTABLISHED',
        final_latency_upper='NOT ESTABLISHED', core_admission=False)
    path.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
