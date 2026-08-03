from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

from vortex_runtime.host_indexed_decision_vm import (
    FORMAT_COMPACT40,
    DecisionVMBuildReport,
    DecisionVMReader,
    build_decision_vm_file,
)


@dataclass(frozen=True)
class BehaviorPath:
    prompt_id: str
    token_ids: tuple[int, ...]
    split: str
    duplicate_of: str | None = None


@dataclass(frozen=True)
class SuffixDepthPoint:
    remaining_tokens: int
    raw_states: int
    unique_suffixes: int
    merged_states: int
    compression_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QuotientFrontierPoint:
    horizon: int
    prompt_paths: int
    raw_path_records: int
    quotient_nodes: int
    merged_records: int
    compression_fraction: float
    unique_full_continuations: int
    start_router_entries: int
    depth_frontier: tuple[SuffixDepthPoint, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["depth_frontier"] = [
            point.to_dict() for point in self.depth_frontier
        ]
        return payload


@dataclass(frozen=True)
class ExactFutureDAG:
    horizon: int
    prompt_order: tuple[str, ...]
    start_addresses: tuple[int, ...]
    token_ids: tuple[int, ...]
    next_addresses: tuple[int, ...]
    token_codebook: tuple[int, ...]
    token_codes: tuple[int, ...]
    suffix_to_address: Mapping[tuple[int, ...], int]

    @property
    def node_count(self) -> int:
        return len(self.token_ids)

    def to_vm_source(self):
        return _FutureDAGVMSource(
            values=self.token_codes,
            next_addresses=self.next_addresses,
            starts=self.start_addresses,
            horizon=self.horizon,
        )

    def manifest(self) -> dict:
        return {
            "version": 1,
            "equivalence": "complete_exact_remaining_token_suffix",
            "horizon": self.horizon,
            "prompt_order": list(self.prompt_order),
            "start_addresses": list(self.start_addresses),
            "node_count": self.node_count,
            "token_codebook": list(self.token_codebook),
        }


@dataclass(frozen=True)
class FutureDAGReplay:
    paths: int
    exact_paths: int
    expected_tokens: int
    exact_tokens: int
    mmap_record_reads: int
    all_exact: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HeldoutFutureBodyCoverage:
    prompt_paths: int
    state_denominator: int
    future_suffix_hits: int
    future_suffix_coverage: float
    full_continuation_hits: int
    causal_start_router_hits: int
    causal_start_router_coverage: float
    first_future_suffix_hit_positions: dict[str, int | None]

    def to_dict(self) -> dict:
        return asdict(self)


class _FutureDAGVMSource:
    def __init__(
        self,
        *,
        values: Sequence[int],
        next_addresses: Sequence[int],
        starts: Sequence[int],
        horizon: int,
    ) -> None:
        if len(values) != len(next_addresses):
            raise ValueError("DAG record arrays must align")
        if not values or not starts or horizon <= 0:
            raise ValueError("DAG VM source cannot be empty")
        self.values = tuple(int(value) for value in values)
        self.next_addresses = tuple(
            int(value) for value in next_addresses
        )
        self.starts = tuple(int(value) for value in starts)
        self.config = SimpleNamespace(
            cells=len(self.values),
            chains=len(self.starts),
            steps=horizon,
        )


def load_experiment_045_paths(
    result_path: str | Path,
) -> tuple[dict, list[BehaviorPath], list[BehaviorPath]]:
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    if payload.get("experiment") != "045_bounded_exact_decision_index_compiler":
        raise ValueError("input is not authoritative Experiment 045 evidence")

    diagnostics = {
        item["prompt_id"]: item
        for item in payload["trace_diagnostics"]
    }
    compiled_ids = list(payload["grammar"]["compiled_prompt_ids"])
    heldout_ids = list(payload["grammar"]["heldout_prompt_ids"])
    compiled = [
        BehaviorPath(
            prompt_id=prompt_id,
            token_ids=tuple(
                int(value)
                for value in diagnostics[prompt_id][
                    "continuation_token_ids"
                ]
            ),
            split="compiled",
        )
        for prompt_id in compiled_ids
    ]
    heldout = [
        BehaviorPath(
            prompt_id=prompt_id,
            token_ids=tuple(
                int(value)
                for value in diagnostics[prompt_id][
                    "continuation_token_ids"
                ]
            ),
            split="heldout",
        )
        for prompt_id in heldout_ids
    ]

    for duplicate in payload.get("duplicate_control", []):
        source_id = duplicate["source_prompt_id"]
        source = next(
            path for path in compiled if path.prompt_id == source_id
        )
        compiled.append(
            BehaviorPath(
                prompt_id=duplicate["duplicate_prompt_id"],
                token_ids=source.token_ids,
                split="duplicate_control",
                duplicate_of=source_id,
            )
        )
    return payload, compiled, heldout


def _validate_paths(paths: Sequence[BehaviorPath], horizon: int) -> None:
    if not paths:
        raise ValueError("at least one behavior path is required")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    prompt_ids = [path.prompt_id for path in paths]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("behavior path IDs must be unique")
    for path in paths:
        if len(path.token_ids) < horizon:
            raise ValueError(
                f"path {path.prompt_id} is shorter than horizon"
            )


def depth_compression_frontier(
    paths: Sequence[BehaviorPath],
    *,
    horizon: int,
) -> tuple[SuffixDepthPoint, ...]:
    _validate_paths(paths, horizon)
    points: list[SuffixDepthPoint] = []
    for remaining in range(1, horizon + 1):
        suffixes = {
            path.token_ids[horizon - remaining : horizon]
            for path in paths
        }
        raw = len(paths)
        unique = len(suffixes)
        merged = raw - unique
        points.append(
            SuffixDepthPoint(
                remaining_tokens=remaining,
                raw_states=raw,
                unique_suffixes=unique,
                merged_states=merged,
                compression_fraction=(merged / raw if raw else 0.0),
            )
        )
    return tuple(points)


def build_exact_future_dag(
    paths: Sequence[BehaviorPath],
    *,
    horizon: int,
    codebook_limit: int = 16,
) -> ExactFutureDAG:
    _validate_paths(paths, horizon)
    if not 1 <= codebook_limit <= 16:
        raise ValueError("compact40 codebook limit must be in 1..16")

    signature_to_address: dict[tuple[int, int], int] = {}
    suffix_to_address: dict[tuple[int, ...], int] = {}
    node_tokens: list[int] = []
    node_next: list[int] = []
    starts: list[int] = []

    for path in paths:
        successor = -1
        tokens = path.token_ids[:horizon]
        for position in range(horizon - 1, -1, -1):
            token = int(tokens[position])
            signature = (token, successor)
            address = signature_to_address.get(signature)
            if address is None:
                address = len(node_tokens)
                signature_to_address[signature] = address
                node_tokens.append(token)
                node_next.append(successor)
            suffix_to_address[tuple(tokens[position:])] = address
            successor = address
        starts.append(successor)

    token_codebook = tuple(sorted(set(node_tokens)))
    if len(token_codebook) > codebook_limit:
        raise ValueError(
            "future DAG token codebook exceeds compact40 capacity: "
            f"{len(token_codebook)} > {codebook_limit}"
        )
    token_to_code = {
        token: code for code, token in enumerate(token_codebook)
    }
    token_codes = tuple(token_to_code[token] for token in node_tokens)

    depth = depth_compression_frontier(paths, horizon=horizon)
    expected_nodes = sum(point.unique_suffixes for point in depth)
    if expected_nodes != len(node_tokens):
        raise RuntimeError(
            "DAG node count disagrees with exact unique-suffix accounting"
        )

    return ExactFutureDAG(
        horizon=horizon,
        prompt_order=tuple(path.prompt_id for path in paths),
        start_addresses=tuple(starts),
        token_ids=tuple(node_tokens),
        next_addresses=tuple(node_next),
        token_codebook=token_codebook,
        token_codes=token_codes,
        suffix_to_address=suffix_to_address,
    )


def quotient_frontier_point(
    paths: Sequence[BehaviorPath],
    *,
    horizon: int,
    codebook_limit: int = 16,
) -> QuotientFrontierPoint:
    dag = build_exact_future_dag(
        paths,
        horizon=horizon,
        codebook_limit=codebook_limit,
    )
    raw = len(paths) * horizon
    merged = raw - dag.node_count
    return QuotientFrontierPoint(
        horizon=horizon,
        prompt_paths=len(paths),
        raw_path_records=raw,
        quotient_nodes=dag.node_count,
        merged_records=merged,
        compression_fraction=(merged / raw if raw else 0.0),
        unique_full_continuations=len(
            {path.token_ids[:horizon] for path in paths}
        ),
        start_router_entries=len(paths),
        depth_frontier=depth_compression_frontier(
            paths,
            horizon=horizon,
        ),
    )


def export_future_dag(
    dag: ExactFutureDAG,
    *,
    vm_path: str | Path,
    manifest_path: str | Path,
    metadata: Mapping[str, object],
) -> tuple[DecisionVMBuildReport, int]:
    build = build_decision_vm_file(
        dag.to_vm_source(),
        vm_path,
        flags=FORMAT_COMPACT40,
    )
    manifest = dag.manifest()
    manifest["vm"] = build.to_dict()
    manifest["metadata"] = dict(metadata)
    encoded = json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(encoded)
    return build, len(encoded)


def replay_future_dag(
    dag: ExactFutureDAG,
    paths: Sequence[BehaviorPath],
    *,
    vm_path: str | Path,
) -> FutureDAGReplay:
    path_by_id = {path.prompt_id: path for path in paths}
    exact_paths = 0
    exact_tokens = 0
    mmap_reads = 0
    expected_tokens = len(paths) * dag.horizon

    with DecisionVMReader(vm_path, verify_payload_checksum=True) as reader:
        for chain, prompt_id in enumerate(dag.prompt_order):
            replay = reader.replay(
                chain=chain,
                maximum_steps=dag.horizon,
            )
            decoded = tuple(
                dag.token_codebook[code] for code in replay.tokens
            )
            expected = path_by_id[prompt_id].token_ids[: dag.horizon]
            exact_tokens += sum(
                int(left == right)
                for left, right in zip(decoded, expected)
            )
            mmap_reads += replay.mmap_record_reads
            if decoded == expected:
                exact_paths += 1

    return FutureDAGReplay(
        paths=len(paths),
        exact_paths=exact_paths,
        expected_tokens=expected_tokens,
        exact_tokens=exact_tokens,
        mmap_record_reads=mmap_reads,
        all_exact=(
            exact_paths == len(paths)
            and exact_tokens == expected_tokens
        ),
    )


def evaluate_heldout_future_body(
    dag: ExactFutureDAG,
    heldout_paths: Sequence[BehaviorPath],
) -> HeldoutFutureBodyCoverage:
    denominator = 0
    suffix_hits = 0
    full_hits = 0
    first_positions: dict[str, int | None] = {}

    compiled_prompt_ids = set(dag.prompt_order)
    causal_router_hits = 0
    for path in heldout_paths:
        if len(path.token_ids) < dag.horizon:
            raise ValueError("held-out path is shorter than DAG horizon")
        tokens = path.token_ids[: dag.horizon]
        first_hit: int | None = None
        if tuple(tokens) in dag.suffix_to_address:
            full_hits += 1
        if path.prompt_id in compiled_prompt_ids:
            causal_router_hits += 1
        for step in range(dag.horizon):
            denominator += 1
            suffix = tuple(tokens[step:])
            if suffix in dag.suffix_to_address:
                suffix_hits += 1
                if first_hit is None:
                    first_hit = step
        first_positions[path.prompt_id] = first_hit

    return HeldoutFutureBodyCoverage(
        prompt_paths=len(heldout_paths),
        state_denominator=denominator,
        future_suffix_hits=suffix_hits,
        future_suffix_coverage=(
            suffix_hits / denominator if denominator else 0.0
        ),
        full_continuation_hits=full_hits,
        causal_start_router_hits=causal_router_hits,
        causal_start_router_coverage=(
            causal_router_hits / len(heldout_paths)
            if heldout_paths
            else 0.0
        ),
        first_future_suffix_hit_positions=first_positions,
    )
