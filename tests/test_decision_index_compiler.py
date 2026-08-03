from __future__ import annotations

from pathlib import Path

import pytest

from vortex_runtime.decision_index_compiler import (
    DecisionTrace,
    GrammarPrompt,
    build_exact_decision_graph,
    evaluate_heldout_coverage,
    exact_state_key,
    export_graph_to_compact40,
    graph_growth_frontier,
    load_bounded_grammar,
    replay_compiled_graph,
)


def make_trace(
    prompt_id: str,
    prompt_tokens: tuple[int, ...],
    continuation: tuple[int, ...],
    *,
    split: str = "compiled",
    duplicate_of: str | None = None,
) -> DecisionTrace:
    return DecisionTrace(
        prompt=GrammarPrompt(
            prompt_id=prompt_id,
            text=prompt_id,
            split=split,
            template_index=0,
            symbol="A",
            count=8,
            duplicate_of=duplicate_of,
        ),
        prompt_token_ids=prompt_tokens,
        continuation_token_ids=continuation,
        model_forward_calls=len(continuation),
        elapsed_ns=1,
        eos_position=None,
    )


def test_repository_grammar_has_exact_declared_split() -> None:
    payload, prompts = load_bounded_grammar(
        "experiments/decision_index_compiler_grammar.json"
    )
    compiled = [prompt for prompt in prompts if prompt.split == "compiled"]
    heldout = [prompt for prompt in prompts if prompt.split == "heldout"]
    duplicate = [
        prompt for prompt in prompts if prompt.split == "duplicate_control"
    ]

    assert len(compiled) == 8
    assert len(heldout) == 4
    assert len(duplicate) == 1
    assert payload["horizons"] == [2, 4, 8]
    assert duplicate[0].duplicate_of == "template0-symbolA-count8"
    source = next(
        prompt for prompt in compiled if prompt.prompt_id == duplicate[0].duplicate_of
    )
    assert duplicate[0].text == source.text


def test_exact_state_key_has_unambiguous_prompt_boundary() -> None:
    assert exact_state_key((1, 2), (3,)) == (1, 2, -1, 3)
    assert exact_state_key((1,), (2, 3)) == (1, -1, 2, 3)
    assert exact_state_key((1, 2), (3,)) != exact_state_key((1,), (2, 3))


def test_duplicate_prompt_path_reuses_every_exact_node() -> None:
    source = make_trace("source", (10, 11), (1, 2, 3, 4))
    duplicate = make_trace(
        "duplicate",
        (10, 11),
        (1, 2, 3, 4),
        split="duplicate_control",
        duplicate_of="source",
    )
    other = make_trace("other", (20, 21), (1, 2, 3, 4))

    graph = build_exact_decision_graph(
        [source, duplicate, other],
        horizon=4,
    )
    assert graph.node_count == 8
    assert graph.prompt_start_addresses[0] == graph.prompt_start_addresses[1]
    assert graph.prompt_start_addresses[2] != graph.prompt_start_addresses[0]

    frontier = graph_growth_frontier(
        [source, duplicate, other],
        [2, 4],
    )
    assert frontier[0].path_records == 6
    assert frontier[0].unique_exact_prefix_nodes == 4
    assert frontier[0].exact_duplicate_records_removed == 2
    assert frontier[1].path_records == 12
    assert frontier[1].unique_exact_prefix_nodes == 8
    assert frontier[1].exact_duplicate_records_removed == 4


def test_inconsistent_transition_for_same_exact_prefix_is_rejected() -> None:
    left = make_trace("left", (10,), (1, 2))
    right = make_trace("right", (10,), (9, 2))
    with pytest.raises(ValueError, match="inconsistent"):
        build_exact_decision_graph([left, right], horizon=2)


def test_compact40_codebook_overflow_is_rejected() -> None:
    traces = [
        make_trace(
            f"prompt-{index}",
            (100 + index,),
            (index,),
        )
        for index in range(17)
    ]
    with pytest.raises(ValueError, match="codebook exceeds"):
        build_exact_decision_graph(
            traces,
            horizon=1,
            codebook_limit=16,
        )


def test_graph_exports_and_replays_exactly_through_compact40(
    tmp_path: Path,
) -> None:
    source = make_trace("source", (10, 11), (5, 6, 5, 7))
    duplicate = make_trace(
        "duplicate",
        (10, 11),
        (5, 6, 5, 7),
        split="duplicate_control",
        duplicate_of="source",
    )
    other = make_trace("other", (20, 21), (6, 5, 7, 5))
    traces = [source, duplicate, other]
    graph = build_exact_decision_graph(traces, horizon=4)

    vm_path = tmp_path / "compiled.vtx"
    manifest_path = tmp_path / "compiled.json"
    build, manifest_bytes = export_graph_to_compact40(
        graph,
        vm_path=vm_path,
        manifest_path=manifest_path,
        metadata={"model": "synthetic"},
    )
    replay = replay_compiled_graph(
        graph,
        traces,
        vm_path=vm_path,
    )

    assert build.format_name == "compact40"
    assert build.record_bytes == 5
    assert manifest_bytes == manifest_path.stat().st_size
    assert replay.prompt_paths == 3
    assert replay.exact_paths == 3
    assert replay.exact_tokens == 12
    assert replay.expected_tokens == 12
    assert replay.all_exact


def test_heldout_exact_prefix_coverage_does_not_generalize_semantically() -> None:
    compiled = [
        make_trace("A", (1, 2, 3), (10, 11, 10, 11)),
        make_trace("B", (1, 2, 4), (10, 11, 10, 11)),
    ]
    graph = build_exact_decision_graph(compiled, horizon=4)
    heldout = [
        make_trace(
            "C",
            (1, 2, 5),
            (10, 11, 10, 11),
            split="heldout",
        )
    ]
    coverage = evaluate_heldout_coverage(graph, heldout)

    assert coverage.state_denominator == 4
    assert coverage.compiled_hits == 0
    assert coverage.fallback_tokens == 4
    assert coverage.coverage == 0.0
    assert coverage.first_miss_positions == {"C": 0}
