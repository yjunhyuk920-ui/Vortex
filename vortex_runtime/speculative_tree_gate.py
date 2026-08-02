from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

from vortex_runtime.feasibility import GIB, ModelSpec


@dataclass(frozen=True)
class TreeVerificationBudget:
    hot_bits: int
    tree_nodes: int
    tree_depth: int
    committed_tokens: int
    hot_weight_gib: float
    transfer_seconds_per_tree: float
    compute_seconds_per_tree: float
    ideal_seconds_per_committed_token: float
    serialized_seconds_per_committed_token: float
    baseline_seconds_per_token: float
    allowed_seconds_per_token: float
    minimum_committed_tokens_ideal: int
    minimum_committed_tokens_serialized: int
    depth_can_meet_ideal: bool
    depth_can_meet_serialized: bool
    observed_ideal_pass: bool
    observed_serialized_pass: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def speculative_tree_verification_budget(
    *,
    target: ModelSpec,
    baseline: ModelSpec,
    hot_bits: int,
    tree_nodes: int,
    tree_depth: int,
    committed_tokens: int,
    host_to_device_gib_s: float = 24.0,
    hot_effective_tops: float = 160.0,
    baseline_memory_gib_s: float = 300.0,
    baseline_effective_tflops: float = 40.0,
    target_ratio: float = 1.2,
) -> TreeVerificationBudget:
    """Budget a target-side lower-bound pass over a speculative token tree.

    The selected coarse full-rank representation is streamed once and every
    retained tree node is evaluated in the same pass. Only one contiguous
    root-to-leaf prefix may be committed. The drafter is deliberately charged
    zero cost, and higher precision needed for exact verification is omitted.
    Therefore passing this function is necessary but not sufficient; failing it
    rejects any more expensive exact progressive verifier for the same tree.
    """

    if not 2 <= hot_bits < target.weight_bits:
        raise ValueError("hot_bits must be below target source precision")
    if tree_nodes <= 0 or tree_depth <= 0:
        raise ValueError("tree_nodes and tree_depth must be positive")
    if not 0 <= committed_tokens <= tree_depth:
        raise ValueError("committed_tokens must be in [0, tree_depth]")
    positive = (
        host_to_device_gib_s,
        hot_effective_tops,
        baseline_memory_gib_s,
        baseline_effective_tflops,
        target_ratio,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("hardware and target values must be positive")

    hot_weight_gib = target.parameters * hot_bits / 8 / GIB
    transfer_seconds = hot_weight_gib / host_to_device_gib_s
    target_ops_per_node = (
        target.dense_linear_flops_per_token
        + target.dense_attention_flops_per_token
    )
    compute_seconds = (
        target_ops_per_node * tree_nodes / (hot_effective_tops * 1e12)
    )

    baseline_weight_seconds = baseline.weight_bytes / GIB / baseline_memory_gib_s
    baseline_ops = (
        baseline.dense_linear_flops_per_token
        + baseline.dense_attention_flops_per_token
    )
    baseline_compute_seconds = baseline_ops / (
        baseline_effective_tflops * 1e12
    )
    baseline_seconds = max(baseline_weight_seconds, baseline_compute_seconds)
    allowed_seconds = target_ratio * baseline_seconds

    ideal_tree_seconds = max(transfer_seconds, compute_seconds)
    serialized_tree_seconds = transfer_seconds + compute_seconds
    minimum_ideal = ceil(ideal_tree_seconds / allowed_seconds)
    minimum_serialized = ceil(serialized_tree_seconds / allowed_seconds)

    denominator = max(1, committed_tokens)
    ideal_per_commit = ideal_tree_seconds / denominator
    serialized_per_commit = serialized_tree_seconds / denominator

    return TreeVerificationBudget(
        hot_bits=hot_bits,
        tree_nodes=tree_nodes,
        tree_depth=tree_depth,
        committed_tokens=committed_tokens,
        hot_weight_gib=hot_weight_gib,
        transfer_seconds_per_tree=transfer_seconds,
        compute_seconds_per_tree=compute_seconds,
        ideal_seconds_per_committed_token=ideal_per_commit,
        serialized_seconds_per_committed_token=serialized_per_commit,
        baseline_seconds_per_token=baseline_seconds,
        allowed_seconds_per_token=allowed_seconds,
        minimum_committed_tokens_ideal=minimum_ideal,
        minimum_committed_tokens_serialized=minimum_serialized,
        depth_can_meet_ideal=tree_depth >= minimum_ideal,
        depth_can_meet_serialized=tree_depth >= minimum_serialized,
        observed_ideal_pass=(
            committed_tokens > 0 and ideal_per_commit <= allowed_seconds
        ),
        observed_serialized_pass=(
            committed_tokens > 0 and serialized_per_commit <= allowed_seconds
        ),
    )


def unique_prefix_node_count(sequences: list[tuple[int, ...]]) -> int:
    """Count unique non-empty prefixes retained by a flattened token tree."""

    prefixes: set[tuple[int, ...]] = set()
    for sequence in sequences:
        for length in range(1, len(sequence) + 1):
            prefixes.add(sequence[:length])
    return len(prefixes)


def longest_reference_prefix(
    sequences: list[tuple[int, ...]],
    reference: tuple[int, ...],
) -> int:
    """Return the longest exact target prefix present in any retained branch."""

    best = 0
    for sequence in sequences:
        limit = min(len(sequence), len(reference))
        length = 0
        while length < limit and sequence[length] == reference[length]:
            length += 1
        best = max(best, length)
    return best
