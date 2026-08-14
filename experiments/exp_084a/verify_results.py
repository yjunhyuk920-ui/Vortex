#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from vortex_runtime.causal_bilinear_rank_gate import (
    FactorizedModularSpan,
    PRIMES,
    exact_coefficients_at_pivots,
    factor_pair_sha256,
    first_exact_residual_coordinate,
    summarize_gate,
    verify_fingerprint_witness,
    witness_strings,
)
from vortex_runtime.causal_residual_atlas_oracle import canonical_sha256


ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(output: Path) -> None:
    checksum_path = output / "checksums.sha256"
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            if relative in expected:
                raise ValueError(f"duplicate checksum row: {relative}")
            expected[relative] = digest
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    if set(expected) != actual:
        raise ValueError("checksum file-set mismatch")
    for relative, digest in expected.items():
        if sha256_file(output / relative) != digest:
            raise ValueError(f"checksum mismatch: {relative}")


def write_checksums(output: Path) -> None:
    rows = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )


def load_pair(output: Path, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(output / row["array_file"], allow_pickle=False) as arrays:
        pair = (
            np.ascontiguousarray(arrays["r"], dtype=np.float32),
            np.ascontiguousarray(arrays["u"], dtype=np.float32),
        )
    if factor_pair_sha256(*pair) != row["pair_sha256"]:
        raise ValueError(f"pair hash mismatch: {row['row_id']}")
    return pair


def assert_equal(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"{name} mismatch: {observed!r} != {expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--write-report", action="store_true")
    arguments = parser.parse_args()
    output = Path(arguments.output_dir).resolve()
    model_dir = Path(arguments.model_dir).resolve()
    verify_checksums(output)
    config = load(output / "config.snapshot.json")
    summary = load(output / "summary.json")
    stored_core = load(output / "raw/deterministic_core.json")
    prompt_rows = load_rows(output / "raw/prompt_rows.jsonl")
    build_rows = load_rows(output / "raw/build_rows.jsonl")
    evaluation_rows = load_rows(output / "raw/evaluation_rows.jsonl")
    control_rows = load_rows(output / "raw/control_rows.jsonl")
    weight_path = model_dir / config["checkpoint"]["weight_file"]
    assert_equal(
        "weight SHA",
        sha256_file(weight_path),
        config["checkpoint"]["weight_sha256"],
    )
    for key, registered_key in (
        ("authority", "authority_sha256"),
        ("prompts", "prompts_sha256"),
        ("target_trace", "target_trace_sha256"),
    ):
        path_key = (
            "authority"
            if key == "authority"
            else f"{key}_path"
        )
        section = "preregistration" if key == "authority" else "registered_inputs"
        path = ROOT / config[section][path_key]
        assert_equal(f"{key} SHA", sha256_file(path), config[section][registered_key])

    primes = tuple(int(value) for value in config["gate"]["primes"])
    assert_equal("prime order", primes, PRIMES)
    left_width = int(config["gate"]["projection_output_rows"])
    right_width = int(config["gate"]["projection_input_width"])
    build_spans = {
        prime: FactorizedModularSpan(left_width, right_width, prime)
        for prime in primes
    }
    basis_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    pivots: list[tuple[int, int]] = []
    for row in build_rows:
        pair = load_pair(output, row)
        increments: dict[str, bool] = {}
        ranks: dict[str, int] = {}
        for prime, span in build_spans.items():
            incremented, _ = span.add(*pair)
            increments[str(prime)] = incremented
            ranks[str(prime)] = span.rank
        coefficients = exact_coefficients_at_pivots(basis_pairs, pivots, pair)
        coordinate = first_exact_residual_coordinate(basis_pairs, pair, coefficients)
        accepted = bool(
            coordinate is not None
            and len(basis_pairs) < int(config["gate"]["ledger_dimension"])
        )
        assert_equal(f"{row['row_id']} modular increments", increments, row["modular_increments"])
        assert_equal(f"{row['row_id']} modular ranks", ranks, row["modular_ranks"])
        assert_equal(f"{row['row_id']} coefficients", witness_strings(coefficients), row["exact_coefficients"])
        assert_equal(
            f"{row['row_id']} coordinate",
            list(coordinate) if coordinate is not None else None,
            row["exact_residual_coordinate"],
        )
        assert_equal(f"{row['row_id']} ledger acceptance", accepted, row["ledger_accepted"])
        if accepted:
            basis_pairs.append(pair)
            pivots.append(coordinate)  # type: ignore[arg-type]

    oracle_spans = {
        prime: FactorizedModularSpan(left_width, right_width, prime)
        for prime in primes
    }
    for row in evaluation_rows:
        pair = load_pair(output, row)
        coefficients = exact_coefficients_at_pivots(basis_pairs, pivots, pair)
        coordinate = first_exact_residual_coordinate(basis_pairs, pair, coefficients)
        exact_hit = coordinate is None
        assert_equal(f"{row['row_id']} exact hit", exact_hit, row["exact_hit"])
        assert_equal(f"{row['row_id']} coefficients", witness_strings(coefficients), row["exact_coefficients"])
        assert_equal(
            f"{row['row_id']} coordinate",
            list(coordinate) if coordinate is not None else None,
            row["exact_residual_coordinate"],
        )
        fields: list[int] = []
        if exact_hit:
            for prime in primes:
                try:
                    passed = verify_fingerprint_witness(
                        basis_pairs,
                        pair,
                        coefficients,
                        prime=prime,
                        seeds=config["gate"]["fingerprint_seeds"],
                    )
                except ValueError:
                    continue
                if not passed:
                    raise ValueError(f"fingerprint witness failed: {row['row_id']}")
                fields.append(prime)
        assert_equal(f"{row['row_id']} fingerprint fields", fields, row["fingerprint_fields"])
        membership: dict[str, int] = {}
        for prime in primes:
            span = FactorizedModularSpan(left_width, right_width, prime)
            for basis_pair in basis_pairs:
                span.add(*basis_pair)
            span.add(*pair)
            membership[str(prime)] = span.rank
        assert_equal(f"{row['row_id']} membership ranks", membership, row["membership_modular_ranks"])
        increments: dict[str, bool] = {}
        ranks: dict[str, int] = {}
        for prime, span in oracle_spans.items():
            incremented, _ = span.add(*pair)
            increments[str(prime)] = incremented
            ranks[str(prime)] = span.rank
        assert_equal(f"{row['row_id']} oracle increments", increments, row["oracle_modular_increments"])
        assert_equal(f"{row['row_id']} oracle ranks", ranks, row["oracle_modular_ranks"])

    failures = [
        f"{row['prompt_id']}:{row['control']}"
        for row in control_rows
        if not row["passed"]
    ]
    oracle_rank = max((span.rank for span in oracle_spans.values()), default=0)
    execution_complete = len(evaluation_rows) == int(config["gate"]["expected_evaluation_rows"])
    gate = summarize_gate(
        evaluation_rows,
        control_failures=failures,
        execution_complete=execution_complete,
        oracle_rank_lower_bound=oracle_rank,
        expected_rows=int(config["gate"]["expected_evaluation_rows"]),
        maximum_misses=int(config["gate"]["maximum_exact_misses"]),
        minimum_family_hits=int(config["gate"]["minimum_family_hits"]),
    )
    gate.update(
        {
            "stop_reason": summary["gate"]["stop_reason"],
            "ledger_dimension": len(basis_pairs),
            "ledger_pivots": [list(value) for value in pivots],
            "build_modular_ranks": {
                str(prime): span.rank for prime, span in build_spans.items()
            },
            "oracle_modular_ranks": {
                str(prime): span.rank for prime, span in oracle_spans.items()
            },
            "target_future_token_reads": 0,
        }
    )
    assert_equal("gate", gate, summary["gate"])
    core = {
        "source": summary["source"],
        "registered_inputs": summary["registered_inputs"],
        "prompt_rows": prompt_rows,
        "build_rows": build_rows,
        "evaluation_rows": evaluation_rows,
        "control_rows": control_rows,
        "gate": gate,
    }
    assert_equal("stored deterministic core", core, stored_core)
    core_sha = canonical_sha256(core)
    assert_equal("deterministic core SHA", core_sha, summary["deterministic_core_sha256"])
    report = {
        "experiment": config["experiment"],
        "passed": True,
        "decision": gate["decision"],
        "deterministic_core_sha256": core_sha,
        "model_forward_calls": 0,
        "verified_build_rows": len(build_rows),
        "verified_evaluation_rows": len(evaluation_rows),
        "verified_exact_misses": gate["exact_misses"],
        "verified_oracle_rank_lower_bound": oracle_rank,
    }
    if arguments.write_report:
        dump(output / "verification.json", report)
        write_checksums(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
