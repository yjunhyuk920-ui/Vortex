"""Read-only metadata audit; never reruns or overwrites the historical study."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_100a.run_experiment import SearchResult, search_coverage_diagnostics


def audit():
    source = ROOT / 'results/exp_100a/0348e87fa385630a84043727ddb9a34c032df4ac/result.json'
    raw = source.read_bytes()
    data = json.loads(raw)
    rows = data['search_rows']
    expected = {
        (b, f['rows'], f['columns'])
        for b in data['search_contract']['block_lengths']
        for f in data['target_families']
    }
    searches = {}
    for row in rows:
        key = tuple(row[k] for k in ('block_length', 'rows', 'columns'))
        if key in searches:
            raise ValueError('duplicate historical search identity')
        # Sentinels encode ONLY recorded cardinality, never reconstructed plans.
        searches[key] = SearchResult(
            *key, tuple(row['state_count_by_depth']),
            (None,) * row['direct_structural_plan_count'],
            (None,) * row['oracle_structural_plan_count'],
        )
    coverage = search_coverage_diagnostics(searches, expected)
    report = {
        'historical_path': str(source.relative_to(ROOT)),
        'historical_sha256': hashlib.sha256(raw).hexdigest(),
        'historical_decision_preserved': data['authoritative_decision'],
        'historical_integrity_failures_preserved': data['integrity_failures'],
        'audit_kind': 'metadata_only_no_search_or_model_rerun',
        'coverage': coverage,
        'recorded_direct_10x_pass_blocks': data['direct_ten_x_pass_blocks'],
        'recorded_oracle_10x_pass_blocks': data['free_transform_ten_x_pass_blocks'],
        'new_scientific_promotion': False,
        'all_factorizations_independently_revalidated': False,
        'universal_infeasibility_proved': False,
    }
    assert source.read_bytes() == raw
    return report


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
