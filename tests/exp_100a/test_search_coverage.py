from dataclasses import replace

from experiments.exp_100a.run_experiment import SearchResult, search_coverage_diagnostics

KEY = (16384, 128256, 16384)
EMPTY = SearchResult(*KEY, (96, 96, 96, 96, 0), (), ())


def test_complete_resource_empty_population_is_not_corrupt():
    report = search_coverage_diagnostics({KEY: EMPTY}, {KEY})
    assert report["integrity_failures"] == []
    assert report["empty_direct"][0]["state_count_by_depth"][-1] == 0
    assert len(report["empty_oracle"]) == 1


def test_missing_population_fails_closed():
    report = search_coverage_diagnostics({}, {KEY})
    assert report["integrity_failures"] == ["incomplete_shape_block_search_population"]
    assert report["missing_keys"] == [KEY]


def test_equal_count_wrong_population_is_detected():
    wrong = (8192, KEY[1], KEY[2])
    report = search_coverage_diagnostics({wrong: replace(EMPTY, block_length=8192)}, {KEY})
    assert report["integrity_failures"] == ["incomplete_shape_block_search_population"]
    assert report["unexpected_keys"] == [wrong]


def test_result_identity_cannot_hide_behind_valid_key():
    report = search_coverage_diagnostics({KEY: replace(EMPTY, rows=1)}, {KEY})
    assert report["integrity_failures"] == ["search_result_identity_mismatch"]


def test_partial_resource_exhaustion_stays_diagnostic():
    other = (8192, KEY[1], KEY[2])
    report = search_coverage_diagnostics(
        {KEY: EMPTY, other: SearchResult(*other, (1,), (object(),), (object(),))},
        {KEY, other},
    )
    assert report["integrity_failures"] == []
    assert len(report["empty_direct"]) == 1
    assert len(report["empty_oracle"]) == 1
