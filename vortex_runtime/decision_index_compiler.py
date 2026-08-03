from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

from vortex_runtime.host_indexed_decision_vm import (
    FORMAT_COMPACT40,
    DecisionVMBuildReport,
    DecisionVMReader,
    build_decision_vm_file,
)


@dataclass(frozen=True)
class GrammarPrompt:
    prompt_id: str
    text: str
    split: str
    template_index: int
    symbol: str
    count: int
    duplicate_of: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DecisionTrace:
    prompt: GrammarPrompt
    prompt_token_ids: tuple[int, ...]
    continuation_token_ids: tuple[int, ...]
    model_forward_calls: int
    elapsed_ns: int
    eos_position: int | None

    @property
    def horizon(self) -> int:
        return len(self.continuation_token_ids)

    @property
    def unique_continuation_tokens(self) -> int:
        return len(set(self.continuation_token_ids))

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt.to_dict(),
            "prompt_tokens": len(self.prompt_token_ids),
            "continuation_tokens": list(self.continuation_token_ids),
            "model_forward_calls": self.model_forward_calls,
            "elapsed_ns": self.elapsed_ns,
            "eos_position": self.eos_position,
            "unique_continuation_tokens": self.unique_continuation_tokens,
        }


@dataclass(frozen=True)
class GraphGrowthPoint:
    horizon: int
    prompt_paths: int
    path_records: int
    unique_exact_prefix_nodes: int
    exact_duplicate_records_removed: int
    deduplication_fraction: float
    nodes_per_prompt: float
    new_nodes_since_previous_horizon: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HeldoutCoverage:
    prompt_paths: int
    state_denominator: int
    compiled_hits: int
    fallback_tokens: int
    coverage: float
    first_miss_positions: dict[str, int | None]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CompiledDecisionGraph:
    horizon: int
    prompt_order: tuple[str, ...]
    prompt_start_addresses: tuple[int, ...]
    node_keys: tuple[tuple[int, ...], ...]
    node_token_ids: tuple[int, ...]
    node_next_addresses: tuple[int, ...]
    token_codebook: tuple[int, ...]
    node_codes: tuple[int, ...]
    key_to_address: Mapping[tuple[int, ...], int]

    @property
    def node_count(self) -> int:
        return len(self.node_keys)

    @property
    def start_count(self) -> int:
        return len(self.prompt_start_addresses)

    def to_vm_source(self):
        return _GraphVMSource(
            values=self.node_codes,
            next_addresses=self.node_next_addresses,
            starts=self.prompt_start_addresses,
            chain_steps=self.horizon,
        )

    def manifest(self) -> dict:
        return {
            "version": 1,
            "horizon": self.horizon,
            "node_count": self.node_count,
            "start_count": self.start_count,
            "prompt_order": list(self.prompt_order),
            "prompt_start_addresses": list(self.prompt_start_addresses),
            "token_codebook": list(self.token_codebook),
            "state_key": "exact_prompt_and_generated_token_prefix",
        }


@dataclass(frozen=True)
class VMReplaySummary:
    prompt_paths: int
    exact_paths: int
    exact_tokens: int
    expected_tokens: int
    all_exact: bool
    mmap_record_reads: int

    def to_dict(self) -> dict:
        return asdict(self)


class _GraphVMSource:
    def __init__(
        self,
        *,
        values: Sequence[int],
        next_addresses: Sequence[int],
        starts: Sequence[int],
        chain_steps: int,
    ) -> None:
        if len(values) != len(next_addresses):
            raise ValueError("graph values and successors must align")
        if not values or not starts or chain_steps <= 0:
            raise ValueError("graph VM source cannot be empty")
        self.values = tuple(int(value) for value in values)
        self.next_addresses = tuple(
            int(value) for value in next_addresses
        )
        self.starts = tuple(int(value) for value in starts)
        self.config = SimpleNamespace(
            cells=len(self.values),
            chains=len(self.starts),
            steps=chain_steps,
        )


def grammar_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def load_bounded_grammar(
    path: str | Path,
) -> tuple[dict, list[GrammarPrompt]]:
    grammar_path = Path(path)
    payload = json.loads(grammar_path.read_text(encoding="utf-8"))
    templates = payload.get("templates")
    symbols = payload.get("symbols")
    counts = payload.get("counts")
    split = payload.get("split", {})
    if not isinstance(templates, list) or not templates:
        raise ValueError("grammar templates are missing")
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("grammar symbols are missing")
    if not isinstance(counts, list) or not counts:
        raise ValueError("grammar counts are missing")

    compiled_symbols = set(split.get("compiled_symbols", []))
    heldout_symbols = set(split.get("heldout_symbols", []))
    if compiled_symbols & heldout_symbols:
        raise ValueError("compiled and held-out symbols overlap")
    if compiled_symbols | heldout_symbols != set(symbols):
        raise ValueError("grammar split does not cover every symbol")

    prompts: list[GrammarPrompt] = []
    for template_index, template in enumerate(templates):
        if not isinstance(template, str):
            raise ValueError("grammar template must be text")
        for symbol in symbols:
            prompt_split = (
                "compiled" if symbol in compiled_symbols else "heldout"
            )
            for count in counts:
                count_value = int(count)
                prompt_id = (
                    f"template{template_index}-symbol{symbol}-count{count_value}"
                )
                prompts.append(
                    GrammarPrompt(
                        prompt_id=prompt_id,
                        text=template.format(
                            symbol=symbol,
                            count=count_value,
                        ),
                        split=prompt_split,
                        template_index=template_index,
                        symbol=str(symbol),
                        count=count_value,
                    )
                )

    duplicate = payload.get("duplicate_control", {})
    if duplicate.get("enabled"):
        source_id = str(duplicate.get("source_prompt_id"))
        duplicate_id = str(duplicate.get("prompt_id"))
        matches = [prompt for prompt in prompts if prompt.prompt_id == source_id]
        if len(matches) != 1:
            raise ValueError("duplicate-control source was not found once")
        source = matches[0]
        prompts.append(
            GrammarPrompt(
                prompt_id=duplicate_id,
                text=source.text,
                split="duplicate_control",
                template_index=source.template_index,
                symbol=source.symbol,
                count=source.count,
                duplicate_of=source.prompt_id,
            )
        )

    prompt_ids = [prompt.prompt_id for prompt in prompts]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("grammar prompt IDs are not unique")
    return payload, prompts


def exact_state_key(
    prompt_token_ids: Sequence[int],
    generated_prefix: Sequence[int],
) -> tuple[int, ...]:
    # -1 cannot be a tokenizer ID and makes the boundary unambiguous.
    return tuple(int(value) for value in prompt_token_ids) + (-1,) + tuple(
        int(value) for value in generated_prefix
    )


def _trace_map(traces: Iterable[DecisionTrace]) -> dict[str, DecisionTrace]:
    result: dict[str, DecisionTrace] = {}
    for trace in traces:
        prompt_id = trace.prompt.prompt_id
        if prompt_id in result:
            raise ValueError(f"duplicate trace for {prompt_id}")
        result[prompt_id] = trace
    return result


def build_exact_decision_graph(
    traces: Sequence[DecisionTrace],
    *,
    horizon: int,
    codebook_limit: int = 16,
) -> CompiledDecisionGraph:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if codebook_limit <= 0 or codebook_limit > 16:
        raise ValueError("compact40 codebook limit must be within 1..16")
    if not traces:
        raise ValueError("at least one trace is required")

    node_specs: dict[
        tuple[int, ...],
        tuple[int, tuple[int, ...] | None],
    ] = {}
    prompt_start_keys: list[tuple[int, ...]] = []
    prompt_order: list[str] = []

    for trace in traces:
        if trace.horizon < horizon:
            raise ValueError(
                f"trace {trace.prompt.prompt_id} is shorter than horizon"
            )
        prompt_order.append(trace.prompt.prompt_id)
        start_key = exact_state_key(trace.prompt_token_ids, ())
        prompt_start_keys.append(start_key)
        for step in range(horizon):
            key = exact_state_key(
                trace.prompt_token_ids,
                trace.continuation_token_ids[:step],
            )
            successor = (
                exact_state_key(
                    trace.prompt_token_ids,
                    trace.continuation_token_ids[: step + 1],
                )
                if step + 1 < horizon
                else None
            )
            token = int(trace.continuation_token_ids[step])
            specification = (token, successor)
            existing = node_specs.get(key)
            if existing is not None and existing != specification:
                raise ValueError(
                    "one exact prefix produced inconsistent transitions"
                )
            node_specs[key] = specification

    ordered_keys = tuple(sorted(node_specs))
    key_to_address = {
        key: address for address, key in enumerate(ordered_keys)
    }
    token_codebook = tuple(
        sorted({specification[0] for specification in node_specs.values()})
    )
    if len(token_codebook) > codebook_limit:
        raise ValueError(
            "compiled token codebook exceeds compact40 capacity: "
            f"{len(token_codebook)} > {codebook_limit}"
        )
    token_to_code = {
        token: code for code, token in enumerate(token_codebook)
    }

    node_token_ids: list[int] = []
    node_codes: list[int] = []
    node_next_addresses: list[int] = []
    for key in ordered_keys:
        token, successor = node_specs[key]
        node_token_ids.append(token)
        node_codes.append(token_to_code[token])
        node_next_addresses.append(
            -1 if successor is None else key_to_address[successor]
        )

    starts = tuple(key_to_address[key] for key in prompt_start_keys)
    return CompiledDecisionGraph(
        horizon=horizon,
        prompt_order=tuple(prompt_order),
        prompt_start_addresses=starts,
        node_keys=ordered_keys,
        node_token_ids=tuple(node_token_ids),
        node_next_addresses=tuple(node_next_addresses),
        token_codebook=token_codebook,
        node_codes=tuple(node_codes),
        key_to_address=key_to_address,
    )


def graph_growth_frontier(
    traces: Sequence[DecisionTrace],
    horizons: Sequence[int],
    *,
    codebook_limit: int = 16,
) -> list[GraphGrowthPoint]:
    if not horizons:
        raise ValueError("at least one horizon is required")
    points: list[GraphGrowthPoint] = []
    previous_nodes = 0
    for horizon in sorted(set(int(value) for value in horizons)):
        graph = build_exact_decision_graph(
            traces,
            horizon=horizon,
            codebook_limit=codebook_limit,
        )
        path_records = len(traces) * horizon
        removed = path_records - graph.node_count
        points.append(
            GraphGrowthPoint(
                horizon=horizon,
                prompt_paths=len(traces),
                path_records=path_records,
                unique_exact_prefix_nodes=graph.node_count,
                exact_duplicate_records_removed=removed,
                deduplication_fraction=(
                    removed / path_records if path_records else 0.0
                ),
                nodes_per_prompt=graph.node_count / len(traces),
                new_nodes_since_previous_horizon=(
                    graph.node_count - previous_nodes
                ),
            )
        )
        previous_nodes = graph.node_count
    return points


def evaluate_heldout_coverage(
    graph: CompiledDecisionGraph,
    heldout_traces: Sequence[DecisionTrace],
) -> HeldoutCoverage:
    denominator = 0
    hits = 0
    first_miss_positions: dict[str, int | None] = {}
    for trace in heldout_traces:
        if trace.horizon < graph.horizon:
            raise ValueError("held-out trace is shorter than graph horizon")
        first_miss: int | None = None
        for step in range(graph.horizon):
            denominator += 1
            key = exact_state_key(
                trace.prompt_token_ids,
                trace.continuation_token_ids[:step],
            )
            if key in graph.key_to_address:
                hits += 1
            elif first_miss is None:
                first_miss = step
        first_miss_positions[trace.prompt.prompt_id] = first_miss
    return HeldoutCoverage(
        prompt_paths=len(heldout_traces),
        state_denominator=denominator,
        compiled_hits=hits,
        fallback_tokens=denominator - hits,
        coverage=(hits / denominator if denominator else 0.0),
        first_miss_positions=first_miss_positions,
    )


def export_graph_to_compact40(
    graph: CompiledDecisionGraph,
    *,
    vm_path: str | Path,
    manifest_path: str | Path,
    metadata: Mapping[str, object],
) -> tuple[DecisionVMBuildReport, int]:
    build = build_decision_vm_file(
        graph.to_vm_source(),
        vm_path,
        flags=FORMAT_COMPACT40,
    )
    manifest = graph.manifest()
    manifest["vm"] = build.to_dict()
    manifest["metadata"] = dict(metadata)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    manifest_path.write_bytes(encoded)
    return build, len(encoded)


def replay_compiled_graph(
    graph: CompiledDecisionGraph,
    traces: Sequence[DecisionTrace],
    *,
    vm_path: str | Path,
) -> VMReplaySummary:
    trace_by_id = _trace_map(traces)
    exact_paths = 0
    exact_tokens = 0
    expected_tokens = len(traces) * graph.horizon
    mmap_reads = 0

    with DecisionVMReader(vm_path, verify_payload_checksum=True) as reader:
        if reader.header.start_count != len(graph.prompt_order):
            raise ValueError("VM start count disagrees with graph")
        for chain_index, prompt_id in enumerate(graph.prompt_order):
            trace = trace_by_id[prompt_id]
            replay = reader.replay(
                chain=chain_index,
                maximum_steps=graph.horizon,
            )
            decoded = tuple(
                graph.token_codebook[code] for code in replay.tokens
            )
            expected = trace.continuation_token_ids[: graph.horizon]
            matching = sum(
                int(left == right)
                for left, right in zip(decoded, expected)
            )
            exact_tokens += matching
            mmap_reads += replay.mmap_record_reads
            if decoded == expected:
                exact_paths += 1

    return VMReplaySummary(
        prompt_paths=len(traces),
        exact_paths=exact_paths,
        exact_tokens=exact_tokens,
        expected_tokens=expected_tokens,
        all_exact=(
            exact_paths == len(traces)
            and exact_tokens == expected_tokens
        ),
        mmap_record_reads=mmap_reads,
    )
