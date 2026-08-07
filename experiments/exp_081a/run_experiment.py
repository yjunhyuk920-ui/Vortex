#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.syndrome_lookup import (
    add,
    aggregate_resource_row,
    compile_random,
    direct_matvec,
    execute_query,
    matvec,
    random_matrix,
    subtract,
)


CONFIG_PATH = ROOT / "experiments/exp_081a/config.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def git_worktree_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    ).strip()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    fraction = index - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def synthetic_controls(config: dict[str, Any]) -> list[dict[str, Any]]:
    prime = int(config["field"]["prime"])
    verification_rows = int(config["field"]["verification_rows"])
    rng = random.Random(8101001)
    rows: list[dict[str, Any]] = []

    def changed_cell(matrix: Any, row: int, column: int, delta: int) -> Any:
        changed = [list(values) for values in matrix]
        changed[row][column] = (changed[row][column] + delta) % prime
        return tuple(tuple(values) for values in changed)

    for trial in range(64):
        m, n, rank = 11, 7, 3
        weight = random_matrix(m, n, prime, rng)
        dictionary = random_matrix(m, rank, prime, rng)
        compiled = compile_random(
            weight=weight,
            dictionary=dictionary,
            verification_rows=verification_rows,
            prime=prime,
            seed=8102000 + trial,
        )
        x = tuple(rng.randrange(prime) for _ in range(n))
        exact = direct_matvec(weight, x, prime)
        coefficients = (1,) + tuple(rng.randrange(prime) for _ in range(rank - 1))
        code_error = matvec(compiled.dictionary, coefficients, prime)
        candidate = subtract(exact, code_error, prime)
        accepted = execute_query(compiled, x, candidate)
        replay = execute_query(compiled, x, candidate)
        rows.append(
            {
                "trial": trial,
                "case": "in_code",
                "verified": accepted.verified,
                "used_fallback": accepted.used_fallback,
                "output_match": accepted.output == exact,
                "replay_match": replay.to_dict() == accepted.to_dict(),
            }
        )
        fault = [0] * m
        fault[trial % m] = 1
        faulty_candidate = add(candidate, fault, prime)
        rejected = execute_query(compiled, x, faulty_candidate)
        rows.append(
            {
                "trial": trial,
                "case": "out_of_code_candidate_fault",
                "verified": rejected.verified,
                "used_fallback": rejected.used_fallback,
                "output_match": rejected.output == exact,
                "replay_match": True,
            }
        )
        nonzero_input = next(index for index, value in enumerate(x) if value)
        unit_delta = pow(x[nonzero_input], prime - 2, prime)
        bad_syndrome = replace(
            compiled,
            recovery_weight=changed_cell(
                compiled.recovery_weight, 0, nonzero_input, unit_delta
            ),
        )
        syndrome_rejected = execute_query(bad_syndrome, x, candidate)
        rows.append(
            {
                "trial": trial,
                "case": "syndrome_fault",
                "verified": syndrome_rejected.verified,
                "used_fallback": syndrome_rejected.used_fallback,
                "output_match": syndrome_rejected.output == exact,
                "replay_match": True,
            }
        )
        bad_dictionary = replace(
            compiled,
            dictionary=changed_cell(compiled.dictionary, 0, 0, 1),
        )
        dictionary_rejected = execute_query(bad_dictionary, x, candidate)
        rows.append(
            {
                "trial": trial,
                "case": "dictionary_fault",
                "verified": dictionary_rejected.verified,
                "used_fallback": dictionary_rejected.used_fallback,
                "output_match": dictionary_rejected.output == exact,
                "replay_match": True,
            }
        )
        bad_fingerprint = replace(
            compiled,
            verification_weight=changed_cell(
                compiled.verification_weight, 0, nonzero_input, unit_delta
            ),
        )
        fingerprint_rejected = execute_query(bad_fingerprint, x, candidate)
        rows.append(
            {
                "trial": trial,
                "case": "fingerprint_fault",
                "verified": fingerprint_rejected.verified,
                "used_fallback": fingerprint_rejected.used_fallback,
                "output_match": fingerprint_rejected.output == exact,
                "replay_match": True,
            }
        )
    singular_rejected = False
    try:
        from vortex_runtime.syndrome_lookup import compile_syndrome_matvec

        compile_syndrome_matvec(
            weight=((1, 2), (3, 4)),
            dictionary=((1,), (0,)),
            recovery=((0, 1),),
            verification=((1, 1),),
            prime=prime,
        )
    except ValueError:
        singular_rejected = True
    rows.append(
        {
            "trial": 64,
            "case": "singular_recovery",
            "verified": False,
            "used_fallback": True,
            "output_match": singular_rejected,
            "replay_match": True,
        }
    )
    return rows


def load_shape_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = ROOT / config["target_shapes"]["path"]
    if sha256_file(path) != config["target_shapes"]["sha256"]:
        raise ValueError("target shape hash mismatch")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def resource_rows(config: dict[str, Any], shapes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gate = config["target_gate"]
    generator = config["generator"]
    rows = []
    for rank in (1, 2, 4, 8, 12, 16):
        row = aggregate_resource_row(
            shapes,
            rank=rank,
            verification_rows=int(config["field"]["verification_rows"]),
            stages=int(generator["stages"]),
            leaves=int(generator["leaves_per_stage"]),
            field_bytes=int(gate["field_sidecar_bytes_per_element"]),
            page_bytes=int(generator["page_dtype_bytes"]),
            target_fraction=float(gate["p50_fraction"]),
            storage_limit_bytes=int(gate["peak_sidecar_bytes"]),
        )
        rows.append(asdict(row))
    return rows


def candidate_features(width: int, count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if count >= width:
        return list(range(width))
    return sorted(rng.sample(range(width), count))


def quantized_page(mean: Any) -> tuple[Any, int]:
    import torch

    maximum = float(mean.abs().max().item()) if mean.numel() else 0.0
    scale = max(1, math.ceil(maximum / 127.0))
    page = torch.round(mean / scale).clamp(-127, 127).to(torch.int8)
    return page, scale


def fit_tree(
    x: Any,
    residual: Any,
    *,
    depth: int,
    feature_count: int,
    minimum_leaf: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    signs = torch.where(
        torch.arange(residual.shape[1]) % 2 == 0,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    )
    proxy = residual.float() @ signs.to(residual.device)
    features = candidate_features(int(x.shape[1]), feature_count, seed)
    pages: list[dict[str, Any]] = []

    def leaf(indices: Any) -> dict[str, Any]:
        mean = residual[indices].float().mean(dim=0) if indices.numel() else torch.zeros(residual.shape[1])
        page, scale = quantized_page(mean)
        page_id = len(pages)
        pages.append({"values": page, "scale": scale, "samples": int(indices.numel())})
        return {"leaf": page_id}

    def build(indices: Any, remaining: int, node_seed: int) -> dict[str, Any]:
        if remaining == 0 or int(indices.numel()) < 2 * minimum_leaf:
            return leaf(indices)
        best: tuple[float, int, float, Any, Any] | None = None
        local = features[:]
        random.Random(node_seed).shuffle(local)
        for feature in local:
            values = x[indices, feature].float()
            threshold = float(values.median().item())
            left_mask = values <= threshold
            left, right = indices[left_mask], indices[~left_mask]
            if int(left.numel()) < minimum_leaf or int(right.numel()) < minimum_leaf:
                continue
            left_proxy, right_proxy = proxy[left], proxy[right]
            score = float(
                left_proxy.sum().square().item() / left.numel()
                + right_proxy.sum().square().item() / right.numel()
            )
            if best is None or score > best[0]:
                best = (score, feature, threshold, left, right)
        if best is None:
            return leaf(indices)
        _, feature, threshold, left, right = best
        return {
            "feature": feature,
            "threshold": threshold,
            "left": build(left, remaining - 1, node_seed * 2 + 1),
            "right": build(right, remaining - 1, node_seed * 2 + 2),
        }

    tree = build(torch.arange(x.shape[0]), depth, seed)
    tree["pages"] = pages
    return tree


def apply_tree(tree: dict[str, Any], x: Any) -> Any:
    import torch

    pages = tree["pages"]
    output = torch.empty((x.shape[0], pages[0]["values"].numel()), dtype=torch.float32)
    for row_index in range(x.shape[0]):
        node = tree
        while "leaf" not in node:
            value = float(x[row_index, node["feature"]].item())
            node = node["left"] if value <= node["threshold"] else node["right"]
        page = pages[node["leaf"]]
        output[row_index] = page["values"].float() * int(page["scale"])
    return output


def fit_forest(x: Any, y: Any, config: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]], Any]:
    import torch

    prediction = torch.zeros_like(y, dtype=torch.float32)
    trees = []
    generator = config["generator"]
    for stage in range(int(generator["stages"])):
        residual = y.float() - prediction
        tree = fit_tree(
            x,
            residual,
            depth=int(generator["tree_depth"]),
            feature_count=int(generator["candidate_split_features"]),
            minimum_leaf=int(generator["minimum_leaf_samples"]),
            seed=seed + stage * 1009,
        )
        prediction += apply_tree(tree, x)
        trees.append(tree)
    return trees, prediction


def apply_forest(trees: list[dict[str, Any]], x: Any) -> Any:
    import torch

    width = trees[0]["pages"][0]["values"].numel()
    prediction = torch.zeros((x.shape[0], width), dtype=torch.float32)
    for tree in trees:
        prediction += apply_tree(tree, x)
    return prediction


def modular_basis(vectors: Any, rank: int, prime: int) -> list[Any]:
    import numpy as np

    basis: list[tuple[int, Any]] = []
    for source in vectors:
        value = np.mod(source.astype(np.int64), prime)
        for pivot, row in basis:
            factor = int(value[pivot])
            if factor:
                value = np.mod(value - factor * row, prime)
        nonzero = np.flatnonzero(value)
        if not len(nonzero):
            continue
        pivot = int(nonzero[0])
        inverse = pow(int(value[pivot]), prime - 2, prime)
        value = np.mod(value * inverse, prime)
        # Keep discovery order. Each later row is reduced against all earlier
        # pivots, which makes the same forward elimination valid at query time.
        basis.append((pivot, value))
        if len(basis) == rank:
            break
    return basis


def in_modular_span(vector: Any, basis: list[Any], prime: int) -> bool:
    import numpy as np

    value = np.mod(vector.astype(np.int64), prime)
    for pivot, row in basis:
        factor = int(value[pivot])
        if factor:
            value = np.mod(value - factor * row, prime)
    return not bool(np.any(value))


def relative_rows(error: Any, target: Any) -> list[float]:
    import torch

    numerator = torch.linalg.vector_norm(error.float(), dim=1)
    denominator = torch.linalg.vector_norm(target.float(), dim=1).clamp_min(1e-12)
    return [float(x) for x in (numerator / denominator).tolist()]


def resolve_projection(layer: Any, projection: str) -> Any:
    if projection == "down_proj":
        return layer.mlp.down_proj
    attention = getattr(layer, "self_attn", None)
    if attention is None:
        attention = getattr(layer, "linear_attn", None)
    if attention is None or not hasattr(attention, "q_proj"):
        raise ValueError("registered q_proj module is unavailable")
    return attention.q_proj


def capture_real_inputs(config: dict[str, Any]) -> tuple[dict[Any, Any], dict[str, Any], dict[Any, Any]]:
    import torch
    from experiments.exp_077a.run_experiment import load_target, tokenize_prompt

    model_path = ROOT / config["checkpoint"]["local_path"]
    weight_path = model_path / "model.safetensors-00001-of-00001.safetensors"
    prompt_path = ROOT / config["prompts"]["path"]
    if sha256_file(weight_path) != config["checkpoint"]["weight_sha256"]:
        raise ValueError("checkpoint weight hash mismatch")
    if sha256_file(prompt_path) != config["prompts"]["sha256"]:
        raise ValueError("prompt hash mismatch")
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    build_ids = [row["id"] for row in prompts["build"]]
    evaluation_ids = [row["id"] for row in prompts["evaluation"]]
    build_families = {row["family"] for row in prompts["build"]}
    evaluation_families = {row["family"] for row in prompts["evaluation"]}
    required_families = set(config["prompts"]["required_families"])
    prompt_audit = {
        "build_count": len(build_ids),
        "evaluation_count": len(evaluation_ids),
        "build_duplicate_ids": len(build_ids) - len(set(build_ids)),
        "evaluation_duplicate_ids": len(evaluation_ids) - len(set(evaluation_ids)),
        "build_evaluation_overlap": len(set(build_ids) & set(evaluation_ids)),
        "build_missing_families": sorted(required_families - build_families),
        "evaluation_missing_families": sorted(required_families - evaluation_families),
    }
    if any(
        prompt_audit[key]
        for key in (
            "build_duplicate_ids",
            "evaluation_duplicate_ids",
            "build_evaluation_overlap",
            "build_missing_families",
            "evaluation_missing_families",
        )
    ):
        raise ValueError(f"prompt split audit failed: {prompt_audit}")
    target, tokenizer, model_info = load_target(model_path)
    model_info = {**model_info, "prompt_audit": prompt_audit}
    captures: dict[Any, dict[str, list[Any]]] = {}
    weights: dict[Any, Any] = {}
    current: dict[str, str] = {}
    handles = []
    for layer_index in config["real_gate"]["layer_indices"]:
        layer = target.model.layers[int(layer_index)]
        for projection in config["real_gate"]["projection_names"]:
            module = resolve_projection(layer, projection)
            key = (int(layer_index), projection)
            captures[key] = {"build_x": [], "evaluation_x": [], "build_meta": [], "evaluation_meta": []}
            weights[key] = module.weight.detach().float().cpu().contiguous()

            def hook(_module: Any, args: Any, *, _key=key) -> None:
                value = args[0].detach().float().reshape(-1, args[0].shape[-1]).cpu()
                split = current["split"]
                captures[_key][f"{split}_x"].append(value)
                captures[_key][f"{split}_meta"].extend(
                    {"prompt_id": current["id"], "family": current["family"]}
                    for _ in range(value.shape[0])
                )

            handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for split in ("build", "evaluation"):
                for row in prompts[split]:
                    current.update(split=split, id=row["id"], family=row["family"])
                    token_ids = tokenize_prompt(
                        tokenizer, row["prompt"], max_tokens=int(config["prompts"]["max_prompt_tokens"])
                    )
                    target(input_ids=token_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    packed = {}
    for key, value in captures.items():
        packed[key] = {
            "build_x": torch.cat(value["build_x"], dim=0),
            "evaluation_x": torch.cat(value["evaluation_x"], dim=0),
            "build_meta": value["build_meta"],
            "evaluation_meta": value["evaluation_meta"],
        }
    return packed, model_info, weights


def real_projection_gate(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import torch

    captures, model_info, weights = capture_real_inputs(config)
    projection_rows: list[dict[str, Any]] = []
    family_values: dict[str, list[tuple[bool, float, int]]] = defaultdict(list)
    rank = int(config["field"]["recovery_rank"])
    prime = int(config["field"]["prime"])
    for index, key in enumerate(sorted(captures)):
        layer_index, projection = key
        capture = captures[key]
        build_x, eval_x = capture["build_x"], capture["evaluation_x"]
        weight = weights[key]
        weight_scale = max(float(weight.abs().max().item()) / 7.0, 1e-12)
        weight_q = torch.round(weight / weight_scale).clamp(-7, 7).float()
        activation_scale = max(float(build_x.abs().max().item()) / 127.0, 1e-12)
        build_q = torch.round(build_x / activation_scale).clamp(-127, 127).float()
        eval_q = torch.round(eval_x / activation_scale).clamp(-127, 127).float()
        build_y = build_q @ weight_q.T
        eval_y = eval_q @ weight_q.T
        trees, build_candidate = fit_forest(build_q, build_y, config, 8103000 + index * 10000)
        eval_candidate = apply_forest(trees, eval_q)
        build_residual = build_y - build_candidate
        eval_residual = eval_y - eval_candidate
        _, _, vh = torch.linalg.svd(build_residual.float(), full_matrices=False)
        favorable_basis = vh[: min(rank, vh.shape[0])].T.contiguous()
        corrected = eval_residual - (eval_residual @ favorable_basis) @ favorable_basis.T
        generator_relative = relative_rows(eval_residual, eval_y)
        corrected_relative = relative_rows(corrected, eval_y)
        integer_build = torch.round(build_residual).to(torch.int64).numpy()
        integer_eval = torch.round(eval_residual).to(torch.int64).numpy()
        basis = modular_basis(integer_build, rank, prime)
        exact_membership = [in_modular_span(row, basis, prime) for row in integer_eval]
        parameters = int(weight.numel())
        for meta, exact, error in zip(capture["evaluation_meta"], exact_membership, corrected_relative):
            family_values[meta["family"]].append((exact, error, parameters))
        projection_rows.append(
            {
                "layer_index": layer_index,
                "projection": projection,
                "input_width": int(weight.shape[1]),
                "output_width": int(weight.shape[0]),
                "parameters": parameters,
                "build_samples": int(build_x.shape[0]),
                "evaluation_samples": int(eval_x.shape[0]),
                "weight_scale": weight_scale,
                "activation_scale": activation_scale,
                "activation_saturation_fraction": float((eval_q.abs() == 127).float().mean().item()),
                "tree_leaf_counts": [len(tree["pages"]) for tree in trees],
                "modular_dictionary_rank": len(basis),
                "exact_coverage": sum(exact_membership) / len(exact_membership),
                "generator_relative_l2_p50": percentile(generator_relative, 0.5),
                "generator_relative_l2_p95": percentile(generator_relative, 0.95),
                "corrected_relative_l2_p50": percentile(corrected_relative, 0.5),
                "corrected_relative_l2_p95": percentile(corrected_relative, 0.95),
            }
        )
    family_rows = []
    for family, values in sorted(family_values.items()):
        total_weight = sum(weight for _, _, weight in values)
        exact_weight = sum(weight for exact, _, weight in values if exact)
        errors = [error for _, error, _ in values]
        family_rows.append(
            {
                "family": family,
                "samples": len(values),
                "weighted_exact_coverage": exact_weight / total_weight,
                "corrected_relative_l2_p50": percentile(errors, 0.5),
                "corrected_relative_l2_p95": percentile(errors, 0.95),
            }
        )
    total_weight = sum(row["parameters"] * row["evaluation_samples"] for row in projection_rows)
    exact_weight = sum(
        row["parameters"] * row["evaluation_samples"] * row["exact_coverage"]
        for row in projection_rows
    )
    all_corrected = []
    for values in family_values.values():
        all_corrected.extend(error for _, error, _ in values)
    aggregate = {
        "model_info": model_info,
        "projection_count": len(projection_rows),
        "build_evaluation_overlap": model_info["prompt_audit"]["build_evaluation_overlap"],
        "weighted_exact_coverage": exact_weight / total_weight,
        "corrected_relative_l2_p50": percentile(all_corrected, 0.5),
        "corrected_relative_l2_p95": percentile(all_corrected, 0.95),
    }
    return projection_rows, family_rows, aggregate


def write_checksums(output: Path) -> None:
    paths = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    text = "".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n" for path in paths)
    (output / "checksums.sha256").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_worktree_clean = git_worktree_clean()
    if not source_worktree_clean:
        raise SystemExit("refusing canonical run from a dirty tracked worktree")
    started = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    controls = synthetic_controls(config)
    shapes = load_shape_rows(config)
    resources = resource_rows(config, shapes)
    projection_rows, family_rows, real = real_projection_gate(config)
    control_pass = all(
        row["output_match"]
        and row["replay_match"]
        and (row["verified"] if row["case"] == "in_code" else row["used_fallback"])
        for row in controls
    )
    authoritative_resource = next(row for row in resources if row["rank"] == config["field"]["recovery_rank"])
    gate = config["target_gate"]
    resource_pass = bool(
        authoritative_resource["operation_pass"]
        and authoritative_resource["traffic_pass"]
        and authoritative_resource["storage_pass"]
    )
    coverage_pass = real["weighted_exact_coverage"] >= gate["required_exact_coverage"]
    observed_families = {row["family"] for row in family_rows}
    required_families = set(config["prompts"]["required_families"])
    family_pass = required_families <= observed_families and all(
        row["weighted_exact_coverage"] >= gate["required_family_exact_coverage"]
        for row in family_rows
    )
    quality_pass = bool(
        real["corrected_relative_l2_p50"] <= gate["maximum_corrected_relative_l2_p50"]
        and real["corrected_relative_l2_p95"] <= gate["maximum_corrected_relative_l2_p95"]
    )
    if not control_pass:
        decision = config["decisions"]["control_failure"]
    elif resource_pass and coverage_pass and family_pass and quality_pass:
        decision = config["decisions"]["pass"]
    else:
        decision = config["decisions"]["scientific_failure"]
    aggregate = {
        **real,
        "control_pass": control_pass,
        "resource_pass": resource_pass,
        "coverage_pass": coverage_pass,
        "family_pass": family_pass,
        "quality_pass": quality_pass,
        "decision": decision,
        "authoritative_resource": authoritative_resource,
    }
    deterministic_core = {
        "controls": controls,
        "resources": resources,
        "projections": projection_rows,
        "families": family_rows,
        "aggregate": aggregate,
    }
    summary = {
        "experiment": config["experiment"],
        "name": config["name"],
        "phase": ["A-theory", "B-synthetic-reference", "C-small-real-checkpoint-necessary-gate"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "MEASURED": {
            "control_case_count": len(controls),
            "control_failures": sum(not row["output_match"] for row in controls),
            "weighted_exact_coverage": real["weighted_exact_coverage"],
            "corrected_relative_l2_p50": real["corrected_relative_l2_p50"],
            "corrected_relative_l2_p95": real["corrected_relative_l2_p95"],
            "deterministic_core_sha256": canonical_sha256(deterministic_core),
            "wall_seconds": time.perf_counter() - started,
        },
        "DERIVED": {
            "authoritative_resource": authoritative_resource,
            "fingerprint_union_upper_bound": config["field"]["maximum_union_queries"] / (config["field"]["prime"] ** config["field"]["verification_rows"]),
            "gate": {key: aggregate[key] for key in ("control_pass", "resource_pass", "coverage_pass", "family_pass", "quality_pass")},
        },
        "REGISTERED_EXTERNAL_INPUTS": {
            "shape_sha256": config["target_shapes"]["sha256"],
            "prompt_sha256": config["prompts"]["sha256"],
            "weight_sha256": config["checkpoint"]["weight_sha256"],
        },
        "provenance": {
            "source_commit": git_commit(),
            "source_worktree_clean_before_run": source_worktree_clean,
            "config_sha256": sha256_file(CONFIG_PATH),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "claim_boundary": {
            "operation_replacement": "NOT_EXECUTED_MODEL_WIDE",
            "checkpoint": "UNCHANGED_PINNED_PAYLOAD_USED_FOR_ACTIVATION_CAPTURE",
            "integer_projection": "DETERMINISTIC_GLOBAL_SCALE_W4A8_REFERENCE_ONLY",
            "physical_speed": "NOT_MEASURED",
            "target_server": "NO_COMMAND_EXECUTED",
        },
        "UNVERIFIED": [
            "BF16 or production Q4 numerical equivalence",
            "complete Transformer operation replacement and token quality",
            "physical lookup syndrome and fingerprint kernels",
            "peak VRAM SSD PCIe H2D latency and power",
            "122B and arbitrary dense 405B execution",
            "E2 through E7",
        ],
    }
    dump_rows(output / "raw/control_rows.jsonl", controls)
    dump_rows(output / "raw/resource_rows.jsonl", resources)
    dump_rows(output / "raw/projection_rows.jsonl", projection_rows)
    dump_rows(output / "raw/family_rows.jsonl", family_rows)
    dump(output / "raw/input_audit.json", summary["REGISTERED_EXTERNAL_INPUTS"])
    dump(output / "processed/aggregate.json", aggregate)
    dump(output / "summary.json", summary)
    dump(output / "artifacts/environment.json", {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count()})
    (output / "artifacts/contract.txt").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (output / "logs/run.log").write_text(json.dumps({"decision": decision, "aggregate": aggregate}, sort_keys=True) + "\n", encoding="utf-8")
    write_checksums(output)
    print(json.dumps({
        "decision": decision,
        "weighted_exact_coverage": real["weighted_exact_coverage"],
        "corrected_relative_l2_p50": real["corrected_relative_l2_p50"],
        "corrected_relative_l2_p95": real["corrected_relative_l2_p95"],
        "resource": authoritative_resource,
        "core": summary["MEASURED"]["deterministic_core_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
