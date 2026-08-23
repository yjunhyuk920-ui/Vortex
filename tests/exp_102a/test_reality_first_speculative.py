from vortex_runtime.reality_first_speculative import (
    CaseAccounting,
    evaluate_holdout_gate,
    minimum_committed_tokens,
    select_build_block_length,
    verify_greedy_block,
)


def row(case_id: str, k: int, committed: int, latency: float, *, exact: bool = True):
    baseline = 1_000_000
    speculative = int(round(latency * baseline))
    return CaseAccounting(
        case_id=case_id,
        block_length=k,
        accepted_draft_tokens=max(0, committed - 1),
        committed_tokens=committed,
        target_positions_evaluated=k,
        draft_prefill_ns=0,
        draft_decode_ns=speculative // 2,
        bridge_ns=0,
        target_verify_ns=speculative - speculative // 2,
        target_repair_ns=0,
        draft_rollback_ns=0,
        baseline_decode_ns=baseline,
        exact_token_match=exact,
        exact_terminal_state_match=exact,
    )


def test_raw_traffic_requirement_is_not_given_free_compression():
    assert minimum_committed_tokens(target_fraction=0.011851851851851851).minimum_committed_tokens == 85
    assert minimum_committed_tokens(target_fraction=0.014814814814814815).minimum_committed_tokens == 68
    # A measured 1.5x ratio may be reported separately but cannot replace raw.
    assert minimum_committed_tokens(target_fraction=0.011851851851851851, compression_ratio=1.5).minimum_committed_tokens == 57


def test_verifier_commits_exact_mismatch_and_charges_repair():
    outcome = verify_greedy_block([7, 8, 9, 10], [7, 8, 42, 99])
    assert outcome.accepted_draft_tokens == 2
    assert outcome.exact_committed_tokens == (7, 8, 42)
    assert outcome.committed_tokens == 3
    assert outcome.target_positions_evaluated == 4
    assert outcome.target_incremental_repair_calls == 1
    assert outcome.candidate_ratio == 4 / 3


def test_full_acceptance_gets_no_free_bonus_token():
    outcome = verify_greedy_block([1, 2, 3], [1, 2, 3])
    assert outcome.accepted_draft_tokens == 3
    assert outcome.committed_tokens == 3
    assert outcome.target_incremental_repair_calls == 0


def test_build_selection_never_uses_holdout_and_prefers_real_raw_pass():
    build = [
        row("a64", 64, 64, 0.7),
        row("b64", 64, 64, 0.8),
        row("a96", 96, 90, 1.0),
        row("b96", 96, 86, 1.1),
    ]
    assert select_build_block_length(build, raw_required_tokens=85) == 96


def test_holdout_gate_requires_bytes_latency_candidate_ratio_and_exact_state():
    passing = [row("h1", 96, 90, 1.1), row("h2", 96, 86, 1.2)]
    metrics = evaluate_holdout_gate(
        passing,
        selected_block_length=96,
        raw_required_tokens=85,
        p50_latency_limit=1.2,
        p95_latency_limit=1.5,
        p95_candidate_ratio_limit=1.2,
    )
    assert metrics["gate_passed"] is True

    failing = passing + [row("h3", 96, 10, 0.5)]
    metrics = evaluate_holdout_gate(
        failing,
        selected_block_length=96,
        raw_required_tokens=85,
        p50_latency_limit=1.2,
        p95_latency_limit=1.5,
        p95_candidate_ratio_limit=1.2,
    )
    assert metrics["gate_passed"] is False


def test_any_state_mismatch_fails_closed():
    rows = [row("good", 96, 90, 1.0), row("bad", 96, 90, 1.0, exact=False)]
    metrics = evaluate_holdout_gate(
        rows,
        selected_block_length=96,
        raw_required_tokens=85,
        p50_latency_limit=1.2,
        p95_latency_limit=1.5,
        p95_candidate_ratio_limit=1.2,
    )
    assert metrics["exact_token_and_state"] is False
    assert metrics["gate_passed"] is False


def test_cross_tokenizer_bridge_fails_closed_when_committed_prefix_retokenizes():
    from experiments.exp_102a.run_experiment import bridge_completion_to_target

    class UnstableTokenizer:
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            # The combined text changes the already-committed first token.
            return {"input_ids": [9, 2, 3] if text.endswith("x") else [1, 2]}

    suffix, audit = bridge_completion_to_target(
        prompt_text="p",
        completion_text="x",
        target_tokenizer=UnstableTokenizer(),
        target_prompt_ids=[1, 2],
        limit=8,
    )
    assert suffix == []
    assert audit["bridge_prefix_stable"] is False


def test_cross_tokenizer_bridge_returns_only_stable_target_suffix():
    from experiments.exp_102a.run_experiment import bridge_completion_to_target

    class StableTokenizer:
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": [1, 2, 7, 8, 9] if text.endswith("xyz") else [1, 2]}

    suffix, audit = bridge_completion_to_target(
        prompt_text="p",
        completion_text="xyz",
        target_tokenizer=StableTokenizer(),
        target_prompt_ids=[1, 2],
        limit=2,
    )
    assert suffix == [7, 8]
    assert audit["bridge_prefix_stable"] is True
    assert audit["target_suffix_tokens"] == 2


def test_speculative_time_charges_prefill_bridge_and_all_online_components():
    item = CaseAccounting(
        case_id="charged",
        block_length=96,
        accepted_draft_tokens=90,
        committed_tokens=91,
        target_positions_evaluated=97,
        draft_prefill_ns=11,
        draft_decode_ns=13,
        bridge_ns=17,
        target_verify_ns=19,
        target_repair_ns=23,
        draft_rollback_ns=29,
        baseline_decode_ns=200,
        exact_token_match=True,
        exact_terminal_state_match=True,
    )
    assert item.speculative_ns == 112
    assert item.latency_ratio == 112 / 200


def test_direct_draft_sync_crops_and_replays_real_mismatch_token():
    from experiments.exp_102a.run_experiment import synchronize_direct_draft_cache

    class Cache:
        def __init__(self):
            self.length = 100
        def crop(self, length):
            self.length = int(length)
        def __iter__(self):
            return iter(())

    class Model:
        def __init__(self):
            self.tokens = []
        def __call__(self, *, input_ids, past_key_values, use_cache, return_dict):
            token = int(input_ids[0, 0].item())
            self.tokens.append(token)
            past_key_values.length += 1
            return object()

    import torch
    cache = Cache()
    model = Model()
    elapsed, size = synchronize_direct_draft_cache(
        model=model,
        cache=cache,
        prompt_length=7,
        accepted_draft_tokens=3,
        mismatch_token=42,
        torch=torch,
    )
    assert elapsed >= 0
    assert size == 0
    assert cache.length == 11
    assert model.tokens == [42]


def test_run_case_full_acceptance_matches_incremental_terminal_cache():
    import torch
    from types import SimpleNamespace
    from experiments.exp_102a.run_experiment import run_case

    class FakeCache:
        def __init__(self):
            self.tokens = []
        def crop(self, length):
            self.tokens = self.tokens[: int(length)]
        def __iter__(self):
            values = torch.tensor(self.tokens, dtype=torch.float32).view(1, 1, -1, 1)
            yield values, values.clone()

    class FakeTokenizer:
        eos_token_id = None
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            ids = [1, 2]
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}
            return {"input_ids": ids}
        def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            return " ".join(str(int(value)) for value in ids)

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(eos_token_id=None, vocab_size=16)
        def parameters(self):
            return iter(())
        def __call__(self, *, input_ids, past_key_values, use_cache, return_dict):
            rows = input_ids[0].tolist()
            logits = torch.full((1, len(rows), 16), -1000.0)
            for index, token in enumerate(rows):
                past_key_values.tokens.append(int(token))
                logits[0, index, (int(token) + 1) % 16] = 1.0
            return SimpleNamespace(logits=logits)

    tokenizer = FakeTokenizer()
    model = FakeModel()
    accounting, detail = run_case(
        case={"id": "toy", "split": "build", "family": "toy", "prompt": "p"},
        requested_block_length=3,
        arm={"name": "same_family", "direct_token_ids": True, "draft_generation_multiplier": 1},
        target_model=model,
        target_tokenizer=tokenizer,
        draft_model=model,
        draft_tokenizer=tokenizer,
        DynamicCache=FakeCache,
        torch=torch,
    )
    assert accounting.accepted_draft_tokens == 3
    assert accounting.committed_tokens == 3
    assert accounting.exact_token_match is True
    assert accounting.exact_terminal_state_match is True
    assert detail["draft_state_strategy"] == "crop_and_replay_mismatch"
    assert accounting.speculative_ns >= accounting.draft_prefill_ns
