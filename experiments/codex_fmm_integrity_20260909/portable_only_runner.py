from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import io
import json
import math
import os
import platform
from pathlib import Path
import re
try:
    import resource
except ModuleNotFoundError:  # Windows has no POSIX resource module.
    resource = None
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import urllib.error
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.explicit_rectangular_fmm import (  # noqa: E402
    ExactFactorization,
    ExplicitFMMError,
    FamilyCandidate,
    JointPlan,
    SchemeOrientation,
    SequenceState,
    StructuralPlan,
    canonical_sha256,
    exact_factorization,
    execute_factorization,
    expand_beam,
    git_blob_sha1,
    instantiate_family_candidate,
    orientations,
    pareto_filter_orientations,
    prune_family_candidates,
    select_joint_plan,
    evaluate_sequence,
)


@dataclass(frozen=True)
class TensorFamily:
    name: str
    rows: int
    columns: int
    count: int

    @property
    def parameters(self) -> int:
        return self.rows * self.columns * self.count

    @property
    def raw_bf16_bytes(self) -> int:
        return 2 * self.parameters


@dataclass(frozen=True)
class SearchResult:
    block_length: int
    rows: int
    columns: int
    state_count_by_depth: tuple[int, ...]
    direct_structural_plans: tuple[StructuralPlan, ...]
    oracle_structural_plans: tuple[StructuralPlan, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rss_bytes() -> int | None:
    try:
        if resource is None:
            return None  # Unavailable host RSS is not zero or a GPU measurement.
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if platform.system() == "Darwin" else value * 1024
    except Exception:
        return None


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_checksums(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "checksums.sha256").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )


def verify_git_blob(path: Path, expected_sha1: str) -> dict[str, Any]:
    payload = path.read_bytes()
    worktree_bytes = len(payload)
    transport_normalization = "none"
    actual = git_blob_sha1(payload)
    if actual != expected_sha1:
        # Accept only an exactly identified LF Git blob transported as CRLF.
        # The expected digest, content and parsed scientific inputs do not change.
        normalized = payload.replace(b"\r\n", b"\n")
        if git_blob_sha1(normalized) == expected_sha1:
            payload = normalized
            actual = expected_sha1
            transport_normalization = "CRLF_to_registered_LF_blob"
    if actual != expected_sha1:
        raise RuntimeError(
            f"registered Git blob mismatch for {path}: expected {expected_sha1}, got {actual}"
        )
    return {
        "path": str(path.relative_to(ROOT)),
        "git_blob_sha1": actual,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "worktree_bytes": worktree_bytes,
        "transport_normalization": transport_normalization,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "exp-100a-explicit-rectangular-fmm-v1":
        raise RuntimeError("unexpected config schema")
    if not bool(config.get("frozen_before_results")):
        raise RuntimeError("configuration was not frozen before results")
    target = config["target_contract"]
    if int(target["baseline_billions"]) != 4 or int(target["target_billions"]) != 405:
        raise RuntimeError("target inventory changed")
    expected_p50 = (
        float(target["p50_latency_multiple"])
        * float(target["baseline_billions"])
        / float(target["target_billions"])
    )
    expected_p95 = (
        float(target["p95_latency_multiple"])
        * float(target["baseline_billions"])
        / float(target["target_billions"])
    )
    if abs(expected_p50 - float(target["p50_target_fraction"])) > 1e-15:
        raise RuntimeError("registered p50 fraction mismatch")
    if abs(expected_p95 - float(target["p95_target_fraction"])) > 1e-15:
        raise RuntimeError("registered p95 fraction mismatch")
    if float(target["first_core_reduction_fraction"]) != 0.10:
        raise RuntimeError("first 10x Gate changed")
    if int(target["peak_vram_bytes"]) != 8 * 2**30:
        raise RuntimeError("8-GiB Gate changed")

    search = config["search"]
    if int(search["maximum_coefficient_absolute_value"]) != 2:
        raise RuntimeError("small-coefficient scope changed")
    if int(search["minimum_depth"]) != 1:
        raise RuntimeError("minimum recursion depth changed")
    if int(search["maximum_depth"]) < 2:
        raise RuntimeError("maximum recursion depth too small")
    blocks = [int(value) for value in search["block_lengths"]]
    if sorted(set(blocks)) != blocks or min(blocks) < 32:
        raise RuntimeError("block list must be sorted and unique")
    if 128 not in blocks or 16384 not in blocks:
        raise RuntimeError("required comparison blocks missing")
    if float(search["lossless_ratio_grant"]) < 1.0:
        raise RuntimeError("lossless ratio is invalid")
    if [int(value) for value in search["static_scalar_bytes"]] != [2, 4]:
        raise RuntimeError("frozen transformed-word widths changed")


def download_catalog(config: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    source = config["catalog"]
    urls = [str(value) for value in source["urls"]]
    expected_blob = str(source["git_blob_sha1"])
    expected_commit = str(source["commit"])
    if not urls or any(expected_commit not in value for value in urls):
        raise RuntimeError("catalog URL is not commit-pinned")
    errors: list[str] = []
    for url in urls:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Vortex-EXP-100A/1.0"},
            method="GET",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    payload = response.read()
                actual_blob = git_blob_sha1(payload)
                if actual_blob != expected_blob:
                    raise RuntimeError(
                        f"AlphaTensor catalog blob mismatch: expected {expected_blob}, got {actual_blob}"
                    )
                return payload, {
                    "url": url,
                    "repository": source["repository"],
                    "commit": expected_commit,
                    "path": source["path"],
                    "git_blob_sha1": actual_blob,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
                errors.append(f"{url} attempt {attempt + 1}: {error}")
                if attempt < 2:
                    time.sleep(2**attempt)
    raise RuntimeError("catalog download failed: " + " | ".join(errors))


def _unpack_catalog_value(value: np.ndarray) -> tuple[Any, Any, Any] | None:
    if isinstance(value, np.ndarray) and value.dtype == object:
        parts = list(value.tolist())
    else:
        try:
            parts = list(value)
        except TypeError:
            return None
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def load_catalog(
    payload: bytes,
    *,
    maximum_coefficient_absolute_value: int,
    controls_per_factorization: int,
    seed: int,
) -> tuple[
    tuple[ExactFactorization, ...],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    key_pattern = re.compile(r"^(\d+),(\d+),(\d+)")
    exact_rows: list[ExactFactorization] = []
    catalog_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    with np.load(io.BytesIO(payload), allow_pickle=True) as archive:
        for key in sorted(archive.files):
            match = key_pattern.match(str(key))
            if match is None:
                catalog_rows.append(
                    {"key": key, "status": "unsupported_key", "reason": "shape not parseable"}
                )
                continue
            a, b, c = map(int, match.groups())
            unpacked = _unpack_catalog_value(archive[key])
            if unpacked is None:
                catalog_rows.append(
                    {"key": key, "status": "unsupported_value", "reason": "expected three factors"}
                )
                continue
            u, v, w = unpacked
            try:
                factorization = exact_factorization(key, a, b, c, u, v, w)
            except ExplicitFMMError as error:
                catalog_rows.append(
                    {"key": key, "status": "unsupported_nonintegral_or_invalid", "reason": str(error)}
                )
                continue
            manifest = factorization.as_manifest()
            max_abs = max(
                factorization.u_stats.max_abs_coefficient,
                factorization.v_stats.max_abs_coefficient,
                factorization.w_stats.max_abs_coefficient,
            )
            fast = factorization.rank < factorization.classical_rank
            small = max_abs <= maximum_coefficient_absolute_value
            status = "eligible" if fast and small else "exact_but_out_of_scope"
            catalog_rows.append(
                {
                    **manifest,
                    "status": status,
                    "fast_rank": fast,
                    "small_coefficient": small,
                    "maximum_coefficient_absolute_value": max_abs,
                }
            )
            for trial in range(controls_per_factorization):
                left = rng.integers(-3, 4, size=(a, b), dtype=np.int64)
                right = rng.integers(-3, 4, size=(b, c), dtype=np.int64)
                expected = left @ right
                actual = execute_factorization(factorization, left, right)
                control_rows.append(
                    {
                        "key": key,
                        "trial": trial,
                        "input_sha256": canonical_sha256(
                            {"left": left.tolist(), "right": right.tolist()}
                        ),
                        "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                        "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                        "exact": bool(np.array_equal(expected, actual)),
                    }
                )
            if fast and small:
                exact_rows.append(factorization)
    if not exact_rows:
        raise RuntimeError("no eligible exact small-coefficient AlphaTensor factorization")
    return tuple(exact_rows), catalog_rows, control_rows


def load_families(config: Mapping[str, Any]) -> tuple[TensorFamily, ...]:
    path = ROOT / config["registered_inputs"]["target_shapes"]["path"]
    verify_git_blob(
        path, str(config["registered_inputs"]["target_shapes"]["git_blob_sha1"])
    )
    excluded = set(str(value) for value in config["search"]["exclude_tensor_families"])
    rows: list[TensorFamily] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        name = str(value["tensor"])
        if name in excluded:
            continue
        rows.append(
            TensorFamily(
                name=name,
                rows=int(value["rows"]),
                columns=int(value["columns"]),
                count=int(value["count"]),
            )
        )
    if not rows:
        raise RuntimeError("target shape population is empty")
    names = [row.name for row in rows]
    required = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    }
    if set(names) != required:
        raise RuntimeError(f"target family population changed: {sorted(names)}")
    return tuple(rows)


def load_prior(
    config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registered = config["registered_inputs"]
    exp080_path = ROOT / registered["exp_080a"]["path"]
    exp099_path = ROOT / registered["exp_099a"]["path"]
    exp080_identity = verify_git_blob(
        exp080_path, str(registered["exp_080a"]["git_blob_sha1"])
    )
    exp099_identity = verify_git_blob(
        exp099_path, str(registered["exp_099a"]["git_blob_sha1"])
    )
    exp080 = json.loads(exp080_path.read_text(encoding="utf-8"))
    exp099 = json.loads(exp099_path.read_text(encoding="utf-8"))
    if (
        exp080["authoritative_decision"]
        != "REJECT_STANDARD_STRASSEN_HYPERBLOCK_AS_CORE_RETAIN_COST_MODEL_AUXILIARY"
    ):
        raise RuntimeError("EXP-080A decision changed")
    if (
        exp099["authoritative_decision"]
        != "PROMOTE_DYADIC_REASSOCIATION_LOCK_TO_EXPLICIT_RECTANGULAR_FMM_GATE"
    ):
        raise RuntimeError("EXP-099A decision changed")
    if exp099.get("integrity_failures"):
        raise RuntimeError("EXP-099A integrity failure")
    return exp080, exp099, {
        "exp_080a": exp080_identity,
        "exp_099a": exp099_identity,
    }


def repair_fractions(
    exp099: Mapping[str, Any], families: Sequence[TensorFamily]
) -> dict[str, float]:
    stats = exp099["holdout"]["role_stats"]
    fallback = float(exp099["maximum_holdout_role_repair_p95"])
    result: dict[str, float] = {}
    for family in families:
        if family.name in stats:
            result[family.name] = float(stats[family.name]["repair_fraction_p95"])
        else:
            result[family.name] = fallback
    return result


def build_orientations(
    factorizations: Sequence[ExactFactorization], config: Mapping[str, Any]
) -> tuple[SchemeOrientation, ...]:
    raw: list[SchemeOrientation] = []
    for factorization in factorizations:
        raw.extend(orientations(factorization))
    eligible = [
        row
        for row in raw
        if row.rank < row.classical_rank
        and row.max_abs_coefficient
        <= int(config["search"]["maximum_coefficient_absolute_value"])
    ]
    filtered = pareto_filter_orientations(
        eligible,
        maximum_count=int(config["search"]["maximum_orientations"]),
    )
    if not filtered:
        raise RuntimeError("orientation Pareto filter removed every scheme")
    return filtered


def _structural_plan_key(plan: StructuralPlan) -> tuple[Any, ...]:
    return (
        plan.sequence_id,
        plan.cut_depth,
        plan.static_scalar_bytes,
        plan.a_product,
        plan.b_product,
        plan.c_product,
    )


def prune_structural_plans(
    plans: Iterable[StructuralPlan], *, maximum_workspace_bytes: int, maximum_count: int
) -> tuple[StructuralPlan, ...]:
    rows = [row for row in plans if row.favorable_workspace_bytes <= maximum_workspace_bytes]
    unique: dict[tuple[Any, ...], StructuralPlan] = {}
    for row in rows:
        key = _structural_plan_key(row)
        incumbent = unique.get(key)
        if incumbent is None or (
            row.transformed_weight_byte_ratio,
            row.online_operations_without_repair,
            row.favorable_workspace_bytes,
        ) < (
            incumbent.transformed_weight_byte_ratio,
            incumbent.online_operations_without_repair,
            incumbent.favorable_workspace_bytes,
        ):
            unique[key] = row
    rows = list(unique.values())
    rows.sort(
        key=lambda row: (
            row.transformed_weight_byte_ratio,
            row.online_operations_without_repair,
            row.favorable_workspace_bytes,
            row.sequence_id,
            row.cut_depth,
        )
    )
    frontier: list[StructuralPlan] = []
    best_operations = math.inf
    for row in rows:
        if row.online_operations_without_repair < best_operations:
            frontier.append(row)
            best_operations = row.online_operations_without_repair
    if len(frontier) <= maximum_count:
        return tuple(frontier)
    if maximum_count == 1:
        return (frontier[0],)
    indices = {
        round(i * (len(frontier) - 1) / (maximum_count - 1))
        for i in range(maximum_count)
    }
    return tuple(frontier[index] for index in sorted(indices))


def search_shape_block(
    *,
    block_length: int,
    rows: int,
    columns: int,
    schemes: Sequence[SchemeOrientation],
    scheme_map: Mapping[str, SchemeOrientation],
    config: Mapping[str, Any],
) -> SearchResult:
    search = config["search"]
    states: tuple[SequenceState, ...] = (
        SequenceState(
            scheme_ids=(),
            a_product=1,
            b_product=1,
            c_product=1,
            rank_product=1,
            transform_proxy=0,
        ),
    )
    direct_plans: list[StructuralPlan] = []
    oracle_best_by_sequence: dict[str, StructuralPlan] = {}
    state_counts: list[int] = []

    for depth in range(1, int(search["maximum_depth"]) + 1):
        states = expand_beam(
            states=states,
            schemes=schemes,
            m=block_length,
            k=columns,
            n=rows,
            maximum_padding_multiplier=float(search["maximum_padding_multiplier"]),
            per_split_survivors=int(search["per_split_survivors"]),
            beam_width=int(search["beam_width"]),
        )
        state_counts.append(len(states))
        if not states:
            break
        if depth < int(search["minimum_depth"]):
            continue
        for state in states:
            steps = tuple(scheme_map[value] for value in state.scheme_ids)
            oracle = evaluate_sequence(
                m=block_length,
                k=columns,
                n=rows,
                steps=steps,
                cut_depth=0,
                static_scalar_bytes=2,
                activation_bytes=int(search["activation_bytes"]),
                output_bytes=int(search["output_bytes"]),
                accumulator_bytes=int(search["accumulator_bytes"]),
            )
            existing = oracle_best_by_sequence.get(oracle.sequence_id)
            if (
                existing is None
                or oracle.free_transform_operations_without_repair
                < existing.free_transform_operations_without_repair
            ):
                oracle_best_by_sequence[oracle.sequence_id] = oracle

            for cut in range(depth + 1):
                widths = (
                    [2]
                    if cut == 0
                    else [int(value) for value in search["static_scalar_bytes"]]
                )
                for static_bytes in widths:
                    direct_plans.append(
                        evaluate_sequence(
                            m=block_length,
                            k=columns,
                            n=rows,
                            steps=steps,
                            cut_depth=cut,
                            static_scalar_bytes=static_bytes,
                            activation_bytes=int(search["activation_bytes"]),
                            output_bytes=int(search["output_bytes"]),
                            accumulator_bytes=int(search["accumulator_bytes"]),
                        )
                    )
        direct_plans = list(
            prune_structural_plans(
                direct_plans,
                maximum_workspace_bytes=int(config["target_contract"]["peak_vram_bytes"]),
                maximum_count=int(search["maximum_structural_plans_per_shape"]),
            )
        )
        gc.collect()

    oracle_rows = sorted(
        oracle_best_by_sequence.values(),
        key=lambda row: (
            row.free_transform_operations_without_repair,
            row.favorable_workspace_bytes,
            row.sequence_id,
        ),
    )[: int(search["maximum_oracle_plans_per_shape"])]
    return SearchResult(
        block_length=block_length,
        rows=rows,
        columns=columns,
        state_count_by_depth=tuple(state_counts),
        direct_structural_plans=tuple(direct_plans),
        oracle_structural_plans=tuple(oracle_rows),
    )


def shape_searches(
    *,
    families: Sequence[TensorFamily],
    schemes: Sequence[SchemeOrientation],
    config: Mapping[str, Any],
) -> tuple[dict[tuple[int, int, int], SearchResult], list[dict[str, Any]]]:
    scheme_map = {row.scheme_id: row for row in schemes}
    cache: dict[tuple[int, int, int], SearchResult] = {}
    rows: list[dict[str, Any]] = []
    unique_shapes = sorted({(family.rows, family.columns) for family in families})
    for block_length in [int(value) for value in config["search"]["block_lengths"]]:
        for output_rows, input_columns in unique_shapes:
            started = time.perf_counter_ns()
            result = search_shape_block(
                block_length=block_length,
                rows=output_rows,
                columns=input_columns,
                schemes=schemes,
                scheme_map=scheme_map,
                config=config,
            )
            cache[(block_length, output_rows, input_columns)] = result
            row = {
                "block_length": block_length,
                "rows": output_rows,
                "columns": input_columns,
                "state_count_by_depth": list(result.state_count_by_depth),
                "direct_structural_plan_count": len(result.direct_structural_plans),
                "oracle_structural_plan_count": len(result.oracle_structural_plans),
                "best_direct_ratio_without_repair": (
                    min(
                        plan.direct_operation_ratio_without_repair
                        for plan in result.direct_structural_plans
                    )
                    if result.direct_structural_plans
                    else None
                ),
                "best_oracle_ratio_without_repair": (
                    min(
                        plan.free_transform_ratio_without_repair
                        for plan in result.oracle_structural_plans
                    )
                    if result.oracle_structural_plans
                    else None
                ),
                "wall_ns": time.perf_counter_ns() - started,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "phase": "shape_search",
                        "K": block_length,
                        "rows": output_rows,
                        "columns": input_columns,
                        "direct": row["best_direct_ratio_without_repair"],
                        "oracle": row["best_oracle_ratio_without_repair"],
                        "states": row["state_count_by_depth"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return cache, rows


def family_candidates_for_block(
    *,
    block_length: int,
    families: Sequence[TensorFamily],
    repair: Mapping[str, float],
    searches: Mapping[tuple[int, int, int], SearchResult],
    config: Mapping[str, Any],
    free_transforms: bool,
) -> dict[str, tuple[FamilyCandidate, ...]]:
    search = config["search"]
    result: dict[str, tuple[FamilyCandidate, ...]] = {}
    for family in families:
        shape = searches[(block_length, family.rows, family.columns)]
        plans = (
            shape.oracle_structural_plans
            if free_transforms
            else shape.direct_structural_plans
        )
        candidates = [
            instantiate_family_candidate(
                family=family.name,
                rows=family.rows,
                columns=family.columns,
                count=family.count,
                block_length=block_length,
                repair_fraction=float(repair[family.name]),
                plan=plan,
                lossless_ratio=float(search["lossless_ratio_grant"]),
                free_transforms=free_transforms,
            )
            for plan in plans
        ]
        result[family.name] = prune_family_candidates(
            candidates,
            maximum_workspace_bytes=int(config["target_contract"]["peak_vram_bytes"]),
            maximum_candidates=int(search["maximum_family_candidates"]),
        )
    return result


def evaluate_blocks(
    *,
    families: Sequence[TensorFamily],
    repair: Mapping[str, float],
    searches: Mapping[tuple[int, int, int], SearchResult],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = config["target_contract"]
    search = config["search"]
    block_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for block_length in [int(value) for value in search["block_lengths"]]:
        direct_candidates = family_candidates_for_block(
            block_length=block_length,
            families=families,
            repair=repair,
            searches=searches,
            config=config,
            free_transforms=False,
        )
        oracle_candidates = family_candidates_for_block(
            block_length=block_length,
            families=families,
            repair=repair,
            searches=searches,
            config=config,
            free_transforms=True,
        )
        direct = select_joint_plan(
            block_length=block_length,
            plan_kind="explicit_staged_transform",
            candidates_by_family=direct_candidates,
            io_fraction_limit=float(target["p50_target_fraction"]),
            io_bins=int(search["joint_io_bins"]),
        )
        oracle = select_joint_plan(
            block_length=block_length,
            plan_kind="free_transform_rank_oracle",
            candidates_by_family=oracle_candidates,
            io_fraction_limit=float(target["p50_target_fraction"]),
            io_bins=int(search["joint_io_bins"]),
        )

        def summary(plan: JointPlan | None) -> dict[str, Any] | None:
            if plan is None:
                return None
            return {
                "arithmetic_ratio": plan.arithmetic_ratio,
                "io_fraction": plan.io_fraction,
                "maximum_workspace_bytes": plan.maximum_workspace_bytes,
                "total_compiled_static_bytes": plan.total_compiled_static_bytes,
                "ten_x_arithmetic_pass": plan.arithmetic_ratio
                <= float(target["first_core_reduction_fraction"]),
                "p50_arithmetic_pass": plan.arithmetic_ratio
                <= float(target["p50_target_fraction"]),
                "p95_arithmetic_pass": plan.arithmetic_ratio
                <= float(target["p95_target_fraction"]),
                "p50_io_pass": plan.io_fraction
                <= float(target["p50_target_fraction"]),
                "workspace_pass": plan.maximum_workspace_bytes
                <= int(target["peak_vram_bytes"]),
            }

        direct_summary = summary(direct)
        oracle_summary = summary(oracle)
        row = {
            "block_length": block_length,
            "candidate_nodes": block_length,
            "committed_tokens_grant": block_length,
            "candidate_ratio_N_over_A": 1.0,
            "direct": direct_summary,
            "free_transform_oracle": oracle_summary,
        }
        block_rows.append(row)
        if direct is not None:
            selected_rows.extend(
                {"joint_kind": "direct", **candidate.as_dict()}
                for candidate in direct.family_candidates
            )
        if oracle is not None:
            selected_rows.extend(
                {"joint_kind": "free_transform_oracle", **candidate.as_dict()}
                for candidate in oracle.family_candidates
            )
        print(json.dumps({"phase": "joint", **row}, sort_keys=True), flush=True)
    return block_rows, selected_rows


def summarize_prior(
    exp080: Mapping[str, Any], exp099: Mapping[str, Any]
) -> dict[str, Any]:
    best080 = exp080["DERIVED"]["best_constructive_row"]
    return {
        "exp_080a": {
            "decision": exp080["authoritative_decision"],
            "best_block_length": int(best080["block_length"]),
            "best_constructive_ratio": float(best080["constructive_ratio"]),
            "best_workspace_bytes": int(best080["favorable_workspace_bytes"]),
            "first_omega_oracle_pass_K": exp080["DERIVED"][
                "first_omega_oracle_pass_K"
            ],
        },
        "exp_099a": {
            "decision": exp099["authoritative_decision"],
            "holdout_native_order_repair_p50": float(
                exp099["holdout"]["native_order_repair_fraction_p50"]
            ),
            "holdout_native_order_repair_p95": float(
                exp099["holdout"]["native_order_repair_fraction_p95"]
            ),
            "maximum_holdout_role_repair_p95": float(
                exp099["maximum_holdout_role_repair_p95"]
            ),
        },
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_preexisting = {"host.txt", "run.log"}
    unexpected = [
        path.name
        for path in output_dir.iterdir()
        if path.name not in allowed_preexisting
    ]
    if unexpected:
        raise FileExistsError(
            f"refusing to overwrite existing scientific evidence: {unexpected}"
        )

    source_commit = git_commit()
    families = load_families(config)
    exp080, exp099, prior_identities = load_prior(config)
    repair = repair_fractions(exp099, families)
    catalog_payload, catalog_identity = download_catalog(config)
    factorizations, catalog_rows, control_rows = load_catalog(
        catalog_payload,
        maximum_coefficient_absolute_value=int(
            config["search"]["maximum_coefficient_absolute_value"]
        ),
        controls_per_factorization=int(
            config["controls"]["synthetic_trials_per_factorization"]
        ),
        seed=int(config["controls"]["seed"]),
    )
    control_mismatches = [row for row in control_rows if not row["exact"]]
    schemes = build_orientations(factorizations, config)
    scheme_rows = [row.as_dict() for row in schemes]

    searches, search_rows = shape_searches(
        families=families,
        schemes=schemes,
        config=config,
    )
    block_rows, selected_rows = evaluate_blocks(
        families=families,
        repair=repair,
        searches=searches,
        config=config,
    )

    target = config["target_contract"]
    integrity_failures: list[str] = []
    if control_mismatches:
        integrity_failures.append("synthetic_factorization_execution_mismatch")
    if not any(row.get("status") == "eligible" for row in catalog_rows):
        integrity_failures.append("no_eligible_catalog_factorization")
    if not schemes:
        integrity_failures.append("no_pareto_orientation")
    if any(not result.direct_structural_plans for result in searches.values()):
        integrity_failures.append("empty_direct_shape_search")
    if any(not result.oracle_structural_plans for result in searches.values()):
        integrity_failures.append("empty_oracle_shape_search")

    direct_passes = [
        row
        for row in block_rows
        if row["direct"] is not None
        and row["direct"]["ten_x_arithmetic_pass"]
        and row["direct"]["p50_io_pass"]
        and row["direct"]["workspace_pass"]
    ]
    oracle_passes = [
        row
        for row in block_rows
        if row["free_transform_oracle"] is not None
        and row["free_transform_oracle"]["ten_x_arithmetic_pass"]
        and row["free_transform_oracle"]["p50_io_pass"]
        and row["free_transform_oracle"]["workspace_pass"]
    ]
    if integrity_failures:
        decision = "INVALID_EXPLICIT_RECTANGULAR_FMM_CONTROL_FAILURE"
    elif direct_passes:
        decision = (
            "PROMOTE_CATALOGUED_RECTANGULAR_FMM_TO_FINITE_WORD_KERNEL_AND_CAUSAL_BLOCK_GATE"
        )
    elif oracle_passes:
        decision = "RETAIN_RECTANGULAR_RANK_HEADROOM_REQUIRE_EXPLICIT_TRANSFORM_CIRCUIT"
    else:
        decision = "REJECT_CATALOGUED_SMALL_COEFFICIENT_RECTANGULAR_FMM_AS_10X_CORE"

    best_direct = min(
        (row for row in block_rows if row["direct"] is not None),
        key=lambda row: row["direct"]["arithmetic_ratio"],
        default=None,
    )
    best_oracle = min(
        (row for row in block_rows if row["free_transform_oracle"] is not None),
        key=lambda row: row["free_transform_oracle"]["arithmetic_ratio"],
        default=None,
    )
    final_target_passes = [
        row
        for row in direct_passes
        if row["direct"]["p50_arithmetic_pass"]
    ]

    deterministic_core = {
        "schema": config["schema"],
        "mechanism_fingerprint": (
            "official-alphatensor-integral-small-coefficient-catalog/"
            "six-tensor-triangle-orientations/bounded-mixed-recursion-beam/"
            "explicit-leaf-multiply-add-plus-staged-left-weight-output-add-scale-move-ledger/"
            "cut-depth-static-weight-form-stream/original-bf16-or-ideal2-fp32-form-bytes/"
            "exp099-role-p95-native-repair-with-original-row-side-stream/"
            "perfect-N-over-A-one-block-grant"
        ),
        "config_sha256": sha256_file(config_path),
        "source_commit": source_commit,
        "catalog_identity": catalog_identity,
        "prior_identities": prior_identities,
        "prior_evidence": summarize_prior(exp080, exp099),
        "target_families": [
            {
                **asdict(row),
                "parameters": row.parameters,
                "raw_bf16_bytes": row.raw_bf16_bytes,
                "repair_fraction_p95": repair[row.name],
            }
            for row in families
        ],
        "catalog_summary": {
            "catalog_key_count": len(catalog_rows),
            "eligible_factorization_count": len(factorizations),
            "synthetic_control_count": len(control_rows),
            "synthetic_control_mismatch_count": len(control_mismatches),
            "raw_orientation_count": sum(
                len(orientations(row)) for row in factorizations
            ),
            "pareto_orientation_count": len(schemes),
        },
        "search_contract": {
            **config["search"],
            "bounded_search_not_exhaustive": True,
            "candidate_source": "non-deployable perfect future block",
            "candidate_nodes_N": "equal to block length",
            "committed_tokens_A_grant": "equal to block length",
            "native_repair_selector": (
                "perfect EXP-099-style oracle; repaired operations and transformed-checkpoint original-row side stream charged"
            ),
            "offline_static_weight_transform": (
                "free construction; compiled bytes and cold stream charged"
            ),
            "lossless_compression": (
                "same favorable ratio granted to transformed forms and repair rows"
            ),
            "finite_word_transform_exactness": (
                "not established; 2-byte and 4-byte transformed-word modes are favorable capacity screens"
            ),
        },
        "scheme_rows": scheme_rows,
        "search_rows": search_rows,
        "block_rows": block_rows,
        "integrity_failures": integrity_failures,
        "direct_ten_x_pass_blocks": [row["block_length"] for row in direct_passes],
        "free_transform_ten_x_pass_blocks": [
            row["block_length"] for row in oracle_passes
        ],
        "direct_final_p50_pass_blocks": [
            row["block_length"] for row in final_target_passes
        ],
        "best_direct_row": best_direct,
        "best_free_transform_oracle_row": best_oracle,
        "authoritative_decision": decision,
    }
    core_sha = canonical_sha256(deterministic_core)
    result = {
        **deterministic_core,
        "deterministic_core_sha256": core_sha,
        "provenance": {
            "measured": [
                "pinned AlphaTensor catalog bytes and Git blob identity",
                "exact integral tensor reconstruction for every eligible source factorization",
                "deterministic integer execution controls",
                "registered EXP-080A and EXP-099A committed evidence identities",
            ],
            "derived": [
                "mixed recursive split/rank products",
                "leaf multiply/add counts",
                "staged transform add/scale/move counts",
                "cut-depth transformed-checkpoint byte expansion",
                "native-repair original-row side stream",
                "favorable one-pass transform workspace",
                "role-weighted native-order repair",
                "joint p50 I/O-bin dynamic program",
            ],
            "projected": [
                "405B inventory arithmetic and BF16 checkpoint stream",
                "same lossless ratio for checkpoint-static transformed forms",
                "8-GiB favorable workspace screen",
            ],
            "unverified": [
                "causal future activation block",
                "finite-word exact transformed-weight representation",
                "sound selector for native-order repairs",
                "actual packed CPU/GPU kernel",
                "decompression and PCIe/HBM overlap",
                "TARGET-W checkpoint execution",
                "physical 8-GiB allocation including KV and allocator fragmentation",
                "same-machine native 4B Q4 p50/p95",
            ],
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
            "peak_rss_bytes": rss_bytes(),
            "wall_ns": time.perf_counter_ns() - started_ns,
        },
    }

    write_json(output_dir / "artifacts/deterministic_core.json", deterministic_core)
    write_json(output_dir / "raw/catalog_rows.json", catalog_rows)
    write_json(output_dir / "raw/factorization_control_rows.json", control_rows)
    write_json(output_dir / "raw/search_rows.json", search_rows)
    write_json(output_dir / "raw/selected_plan_rows.json", selected_rows)
    write_json(output_dir / "result.json", result)
    write_checksums(output_dir)
    print(
        json.dumps(
            {
                "authoritative_decision": decision,
                "eligible_factorizations": len(factorizations),
                "pareto_orientations": len(schemes),
                "best_direct": best_direct,
                "best_free_transform_oracle": best_oracle,
                "integrity_failures": integrity_failures,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    execute(args.config.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
