from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import resource
import subprocess
import time
from typing import Any, Mapping
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.structured_direct_sum import MultiplicationOracle, catalog_schemes, triple_cyclic_rank, weighted_ratio


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verify_blob(path: Path, expected: str) -> dict[str, Any]:
    data = path.read_bytes()
    actual = git_blob_sha1(data)
    if actual != expected:
        raise RuntimeError(f"blob mismatch {path}: {actual} != {expected}")
    return {"path": str(path.relative_to(ROOT)), "git_blob_sha1": actual, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def validate_config(c: Mapping[str, Any]) -> None:
    if c.get("schema") != "exp-101a-structured-direct-sum-v1" or not c.get("frozen_before_results"):
        raise RuntimeError("invalid frozen config")
    r = c["relation"]
    if [int(x) for x in r["base_shape"]] != [6, 6, 6] or int(r["unit_multiplicity"]) != 137 or int(r["double_multiplicity"]) != 8:
        raise RuntimeError("structured relation changed")
    if int(r["reported_triple_cyclic_rank"]) != 3581065:
        raise RuntimeError("triple cyclic rank changed")
    if float(c["target_contract"]["first_core_reduction_fraction"]) != 0.1:
        raise RuntimeError("10x Gate changed")


def load_families(c: Mapping[str, Any]) -> list[dict[str, int | str]]:
    reg = c["registered_inputs"]["target_shapes"]
    path = ROOT / reg["path"]
    verify_blob(path, reg["git_blob_sha1"])
    excluded = set(c["search"]["exclude_tensor_families"])
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [{"name": r["tensor"], "rows": int(r["rows"]), "columns": int(r["columns"]), "count": int(r["count"])} for r in rows if r["tensor"] not in excluded]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()

    identities = {}
    for key in ("catalog_rows", "exp_100a_result"):
        row = config["registered_inputs"][key]
        identities[key] = verify_blob(ROOT / row["path"], row["git_blob_sha1"])
    catalog = json.loads((ROOT / config["registered_inputs"]["catalog_rows"]["path"]).read_text())
    prior = json.loads((ROOT / config["registered_inputs"]["exp_100a_result"]["path"]).read_text())
    schemes = catalog_schemes(catalog)
    families = load_families(config)

    request = urllib.request.Request(config["relation"]["url"], headers={"User-Agent": "Vortex-EXP-101A/1.0"})
    source = urllib.request.urlopen(request, timeout=90).read()
    source_blob = git_blob_sha1(source)
    source_lines = len([line for line in source.decode("utf-8").splitlines() if line.strip()])
    integrity: list[str] = []
    if source_blob != config["relation"]["git_blob_sha1"]:
        integrity.append("structured_source_blob_mismatch")
    if source_lines != int(config["relation"]["line_count"]):
        integrity.append("structured_source_line_count_mismatch")
    cyclic = triple_cyclic_rank(137, 8, 7)
    if cyclic != int(config["relation"]["reported_triple_cyclic_rank"]):
        integrity.append("triple_cyclic_rank_identity_failure")
    if prior["catalog_summary"]["synthetic_control_mismatch_count"] != 0:
        integrity.append("exp100_exact_factorization_control_failure")

    maximum_depth = int(config["search"]["maximum_depth"])
    state_limit = int(config["search"]["state_limit_per_oracle"])
    # Reuse each exact dynamic-programming cache across every registered block
    # and projection shape. This changes only evaluation order, not the frozen
    # recurrence or candidate language.
    oracles = {
        "catalog_only": MultiplicationOracle(schemes, maximum_depth=maximum_depth, structured_enabled=False, state_limit=state_limit),
        "structured_only": MultiplicationOracle((), maximum_depth=maximum_depth, structured_enabled=True, state_limit=state_limit),
        "catalog_plus_structured": MultiplicationOracle(schemes, maximum_depth=maximum_depth, structured_enabled=True, state_limit=state_limit),
    }
    block_rows = []
    for block_length in config["search"]["block_lengths"]:
        population = [(int(block_length), int(f["columns"]), int(f["rows"]), int(f["count"])) for f in families]
        results = {}
        for name, oracle in oracles.items():
            ratio, candidate, baseline = weighted_ratio(population, oracle)
            results[name] = {"multiplication_ratio": ratio, "multiplications": candidate, "baseline_multiplications": baseline, "cache": oracle.cache_info()}
        row = {"block_length": int(block_length), **results}
        block_rows.append(row)
        print(json.dumps({"K": block_length, "catalog": results["catalog_only"]["multiplication_ratio"], "structured": results["structured_only"]["multiplication_ratio"], "composed": results["catalog_plus_structured"]["multiplication_ratio"]}), flush=True)

    best = min(block_rows, key=lambda row: row["catalog_plus_structured"]["multiplication_ratio"])
    gate = float(config["target_contract"]["first_core_reduction_fraction"])
    if integrity:
        decision = config["decisions"]["control_failure"]
    elif best["catalog_plus_structured"]["multiplication_ratio"] <= gate:
        decision = config["decisions"]["promotion"]
    else:
        decision = config["decisions"]["scientific_rejection"]

    core = {
        "schema": config["schema"],
        "mechanism_fingerprint": "pinned-structured-666-r153/direct-sum-137-unit-plus-8-doubled/cyclic-axis-choice/exact-integral-alphatensor-uniform-composition/classical-fallback/multiplication-only-free-transform-free-byte-free-workspace-oracle",
        "source_identity": {"repository": config["relation"]["repository"], "commit": config["relation"]["commit"], "path": config["relation"]["path"], "git_blob_sha1": source_blob, "sha256": hashlib.sha256(source).hexdigest(), "nonempty_lines": source_lines},
        "triple_cyclic_rank": cyclic,
        "registered_identities": identities,
        "catalog": {"eligible_rows": sum(row.get("status") == "eligible" for row in catalog), "pareto_uniform_orientations": len(schemes), "schemes": [scheme.manifest() for scheme in schemes]},
        "families": families,
        "block_rows": block_rows,
        "best_composed_row": best,
        "ten_x_multiplication_gate": gate,
        "integrity_failures": integrity,
        "authoritative_decision": decision,
        "claim_boundary": config["claim_boundary"],
        "prior_exp100_reclassification": {"recorded_decision": prior["authoritative_decision"], "actual_integrity_except_resource_empty_frontier": prior["catalog_summary"]["synthetic_control_mismatch_count"] == 0, "best_direct_arithmetic_ratio": prior["best_direct_row"]["direct"]["arithmetic_ratio"], "correct_scientific_interpretation": "REJECT_CATALOGUED_SMALL_COEFFICIENT_RECTANGULAR_FMM_AS_10X_CORE"},
    }
    result = {**core, "deterministic_core_sha256": hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "environment": {"python": platform.python_version(), "platform": platform.platform(), "peak_rss_bytes": rss_bytes(), "wall_ns": time.perf_counter_ns() - started, "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()}}
    write_json(output_dir / "artifacts/deterministic_core.json", core)
    write_json(output_dir / "raw/block_rows.json", block_rows)
    write_json(output_dir / "result.json", result)
    checks = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checks.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "checksums.sha256").write_text("\n".join(checks) + "\n")
    print(json.dumps({"decision": decision, "best_K": best["block_length"], "best_ratio": best["catalog_plus_structured"]["multiplication_ratio"], "integrity": integrity}, indent=2))


if __name__ == "__main__":
    main()
